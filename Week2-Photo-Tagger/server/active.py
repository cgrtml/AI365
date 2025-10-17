# server/active.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
from store import Store

class Active:
    """
    Basit bir bellek/k-NN tabanlı "score boost" mekanizması.
    - fit_memory(): accepted=1 + user etiketli örnekleri RAM'e alır
    - boost_scores(scores, kind, query_emb): komşu örneklerin onaylı etiketlerine küçük bir +α ekler
    """

    def __init__(self, store: Store, k: int = 5, alpha: float = 0.08, min_sim: float = 0.20):
        self.store = store
        self.k = k
        self.alpha = alpha
        self.min_sim = min_sim
        self.room_mem: Tuple[np.ndarray, List[set[str]]] = (np.zeros((0,1), np.float32), [])
        self.feat_mem: Tuple[np.ndarray, List[set[str]]] = (np.zeros((0,1), np.float32), [])
        self.fit_memory()

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32, copy=False)
        n = np.linalg.norm(v) + 1e-9
        return v / n

    def fit_memory(self) -> None:
        self.room_mem = self._fetch("room")
        self.feat_mem = self._fetch("feat")

    def _fetch(self, kind: str) -> Tuple[np.ndarray, List[set[str]]]:
        embs, lbl_sets = self.store.get_memory(kind)
        # normalize once
        if embs.shape[0] > 0:
            embs = (embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
        return embs, lbl_sets

    def boost_scores(
        self,
        scores: List[Dict],          # [{"label": str, "score": float}, ...]
        kind: str,                   # 'room' | 'feat'
        query_emb: np.ndarray,
        k: Optional[int] = None,
        alpha: Optional[float] = None,
        renormalize: bool = True,
    ) -> List[Dict]:
        """
        - Bellek boşsa aynen geri döner.
        - Aksi halde, en yakın k komşunun pozitif benzerliğine göre label ağırlıkları hesaplar,
          eşleşen label'ların skorunu +α * weight ile artırır.
        """
        assert kind in ("room", "feat")
        if not scores:
            return scores

        mem_embs, mem_lbls = self.room_mem if kind == "room" else self.feat_mem
        if mem_embs.shape[0] == 0:
            return scores

        k = k or self.k
        alpha = alpha if alpha is not None else self.alpha

        q = self._normalize(query_emb).reshape(-1).astype(np.float32)
        sims = mem_embs @ q  # cosine, çünkü her ikisi de normalize

        # en benzer k komşu
        top_idx = np.argsort(-sims)[:k]
        weights: Dict[str, float] = {}
        total_w = 0.0
        for i in top_idx:
            s = float(sims[i])
            if s < self.min_sim:
                continue
            for label in mem_lbls[i]:
                weights[label] = weights.get(label, 0.0) + s
                total_w += s

        if total_w > 0:
            # normalize etiket ağırlıkları
            for klabel in list(weights.keys()):
                weights[klabel] /= total_w

            # boost uygula
            for item in scores:
                lbl = item["label"]
                if lbl in weights:
                    item["score"] = float(item["score"] + alpha * weights[lbl])

            # (opsiyonel) yeniden normalize
            if renormalize:
                ssum = sum(max(0.0, float(x["score"])) for x in scores)
                if ssum > 0:
                    for x in scores:
                        x["score"] = float(max(0.0, x["score"]) / ssum)

        # skor sıralaması yüksekten düşüğe
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores
