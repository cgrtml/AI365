import os
import io
import base64
from PIL import Image
from dotenv import load_dotenv
from transformers import pipeline

# OpenAI (opsiyonel LLM vision)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

# ---- Global, tek seferlik pipeline (hız için) ----
_OFFLINE_PIPE = None

def _get_offline_pipe():
    """
    Stabil model: vit-gpt2-image-captioning (CPU)
    """
    global _OFFLINE_PIPE
    if _OFFLINE_PIPE is None:
        _OFFLINE_PIPE = pipeline(
            "image-to-text",
            model="nlpconnect/vit-gpt2-image-captioning",
            framework="pt",   # PyTorch
            device=-1         # CPU
        )
    return _OFFLINE_PIPE

# -------- Helpers --------
def _load_image_from_bytes(content: bytes) -> Image.Image:
    return Image.open(io.BytesIO(content)).convert("RGB")

def _to_b64_image(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -------- Offline (HF) --------
def caption_offline(image: Image.Image) -> str:
    pipe = _get_offline_pipe()
    # Normalize et: sabit boyut makul; bazı preprocessing farklarını dengeler
    img = image.convert("RGB").resize((384, 384), Image.BICUBIC)
    out = pipe(img, max_new_tokens=40)  # ⚠️ padding parametresi YOK
    # Çıktı genelde [{'generated_text': '...'}]
    if isinstance(out, list) and len(out) and isinstance(out[0], dict):
        return out[0].get("generated_text", "").strip() or "(no caption)"
    return "(no caption)"

# -------- OpenAI Vision (opsiyonel) --------
def caption_openai(image: Image.Image, model: str = "gpt-4o-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return caption_offline(image)
    client = OpenAI()
    b64 = _to_b64_image(image, fmt="JPEG")
    messages = [
        {"role": "system", "content": "You are a concise image captioning assistant."},
        {"role": "user","content":[
            {"type":"input_text","text":"Give a clear image caption in 1–2 sentences."},
            {"type":"input_image","image_url": f"data:image/jpeg;base64,{b64}"},
        ]},
    ]
    resp = client.chat.completions.create(model=model, temperature=0.2, messages=messages)
    return resp.choices[0].message.content.strip()

# -------- Public Methods (CLI & API) --------
def caption_image(path: str, method: str = "offline") -> str:
    image = Image.open(path).convert("RGB")
    return caption_openai(image) if (method or "offline").lower() == "openai" else caption_offline(image)

def caption_image_from_bytes(content: bytes, method: str = "offline") -> str:
    image = _load_image_from_bytes(content)
    return caption_openai(image) if (method or "offline").lower() == "openai" else caption_offline(image)
