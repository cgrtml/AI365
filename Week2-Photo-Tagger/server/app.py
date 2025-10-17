# server/app.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch, io, yaml, os, hashlib

# Yerel modüller (aynı klasörde)
from store import Store
from active import Active

# ---- Cihaz seçimi
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- FastAPI temel ayar
app = FastAPI(title="Real Estate Photo Tagger", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---- Etiketleri labels.yaml'dan yükle
LABELS_PATH = os.getenv("LABELS_PATH", "labels.yaml")
with open(LABELS_PATH, "r") as f:
    LABELS = yaml.safe_load(f)

ROOM_TYPES: List[str] = LABELS["room_types"]
FEATURES:   List[str] = LABELS["features"]

# ---- CLIP modelini yükle (tek sefer)
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# ---- Store & Active (öğrenme)
store = Store("photo_tagger.db")
active = Active(store)  # k-NN boost belleğini yönetir

# ---- Yardımcılar
def rank_prompts_for_image(image: Image.Image, prompts: List[str]) -> List[Dict]:
    """
    Verilen 'prompts' listesi için CLIP benzerlik skorlarını döndürür (yüksekten düşüğe).
    """
    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image.squeeze(0)  # [num_text]
        probs = logits_per_image.softmax(dim=0)

    scored = [{"label": p, "score": float(s)} for p, s in zip(prompts, probs)]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

def file_sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def image_embedding_only(img: Image.Image):
    """Görsel embedding (normalize) – aktif öğrenme için."""
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.squeeze(0).detach().cpu().numpy().astype("float32")

# ---- Endpoint'ler
@app.get("/")
def root():
    return {"ok": True}

@app.post("/tag")
async def tag_image(
    files: List[UploadFile] = File(...),
    topk_room: int = Query(3, ge=1, le=10),
    topk_feat: int = Query(8, ge=1, le=30),
    minfeat_score: float = Query(0.02, ge=0.0, le=1.0),
):
    """
    Bir veya birden çok görsel yükler; oda türleri ve özellik etiketlerini skorlarıyla döner.
    Ayrıca her görselin sha'sını döner (feedback için).
    """
    results = []
    for uf in files:
        data = await uf.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")

        # 1) Embedding + DB'ye kaydet (öğrenme için)
        emb = image_embedding_only(img)
        sha = file_sha256(data)
        store.upsert_example(sha=sha, embedding=emb)  # accepted/gate_score yoksa None kalır

        # 2) Model skorları
        room_scores = rank_prompts_for_image(img, ROOM_TYPES)[:topk_room]
        feat_all    = rank_prompts_for_image(img, FEATURES)
        feat_scores = [f for f in feat_all if f["score"] >= minfeat_score][:topk_feat]

        # 3) Aktif öğrenme boost'u (kullanıcı onaylı hafızaya göre küçük +α)
        room_scores = active.boost_scores(room_scores, kind="room", query_emb=emb)
        feat_scores = active.boost_scores(feat_scores, kind="feat", query_emb=emb)

        results.append({
            "file": uf.filename,
            "sha": sha,                  # ⟵ önemli: feedback'te kullanılacak
            "room_types": room_scores,
            "features":  feat_scores
        })
    return {"results": results}

@app.post("/feedback")
async def feedback(
    sha: str = Query(..., description="Önceden /tag ile dönen sha"),
    accept: bool = Query(True, description="Görsel uygun mu?"),
    room_types: Optional[List[str]] = Query(None, description="Doğru oda etiketleri (isteğe bağlı, tekrarlı parametre)"),
    features:   Optional[List[str]] = Query(None, description="Doğru feature etiketleri (isteğe bağlı, tekrarlı parametre)"),
):
    """
    Kullanıcı geri bildirimi: kabul/red + düzeltme etiketleri.
    Bu çağrıdan sonra k-NN belleği güncellenir; benzer görsellerde skorlar iyileşir.
    """
    # kabul/red bilgisi
    store.set_accept(sha, accept)
    # düzeltme etiketleri
    if room_types:
        for r in room_types:
            store.add_label(sha, r, "room", "user", score=1.0)
    if features:
        for f in features:
            store.add_label(sha, f, "feat", "user", score=1.0)

    # bellek güncelle (aktif öğrenme)
    active.fit_memory()
    return {"ok": True}
