# server/store.py
from __future__ import annotations
import sqlite3
from typing import Iterable, List, Optional, Tuple, Dict
import numpy as np
import time

class Store:
    """
    Basit SQLite saklama katmanı.
    Tablo/Alanlar:
      - examples(id, sha256 UNIQUE, embedding BLOB, dim INT, gate_score REAL, accepted INT NULL, created_at)
      - labels(id, example_id, label, kind{'room'|'feat'}, source{'user'|'model'}, score REAL, created_at)
    """

    def __init__(self, path: str = "photo_tagger.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    # ---------- helpers ----------
    @staticmethod
    def _np_to_blob(arr: np.ndarray) -> bytes:
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return arr.tobytes(order="C")

    @staticmethod
    def _blob_to_np(blob: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32, count=dim)
        return arr

    # ---------- schema ----------
    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS examples (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              sha256      TEXT NOT NULL UNIQUE,
              embedding   BLOB NOT NULL,
              dim         INTEGER NOT NULL,
              gate_score  REAL,
              accepted    INTEGER,                 -- NULL: bilinmiyor, 0: hayır, 1: evet
              created_at  INTEGER NOT NULL
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_examples_accepted ON examples(accepted);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_examples_sha ON examples(sha256);")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS labels (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              example_id  INTEGER NOT NULL,
              label       TEXT NOT NULL,
              kind        TEXT NOT NULL CHECK(kind IN ('room','feat')),
              source      TEXT NOT NULL CHECK(source IN ('user','model')),
              score       REAL NOT NULL,
              created_at  INTEGER NOT NULL,
              UNIQUE(example_id, label, kind, source),
              FOREIGN KEY(example_id) REFERENCES examples(id) ON DELETE CASCADE
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_labels_example ON labels(example_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_labels_kind_label ON labels(kind, label);")
        self.conn.commit()

    # ---------- examples ----------
    def upsert_example(
        self,
        sha: str,
        embedding: np.ndarray,
        gate_score: Optional[float] = None,
        accepted: Optional[bool] = None,
    ) -> int:
        """
        Yoksa ekler, varsa günceller. example_id döner.
        """
        embedding = embedding.astype(np.float32, copy=False)
        dim = int(embedding.shape[0])
        now = int(time.time())
        cur = self.conn.cursor()

        cur.execute("SELECT id FROM examples WHERE sha256=?;", (sha,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO examples (sha256, embedding, dim, gate_score, accepted, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?);",
                (sha, self._np_to_blob(embedding), dim, gate_score, None if accepted is None else int(accepted), now)
            )
            ex_id = cur.lastrowid
        else:
            ex_id = int(row["id"])
            # embedding'i de güncelleyelim; gate_score/accepted geldiyse set edelim
            sets, vals = [], []
            sets.append("embedding=?"); vals.append(self._np_to_blob(embedding))
            sets.append("dim=?");       vals.append(dim)
            if gate_score is not None:
                sets.append("gate_score=?"); vals.append(gate_score)
            if accepted is not None:
                sets.append("accepted=?");   vals.append(int(accepted))
            sql = f"UPDATE examples SET {', '.join(sets)} WHERE id=?;"
            vals.append(ex_id)
            cur.execute(sql, tuple(vals))

        self.conn.commit()
        return ex_id

    def set_accept(self, sha: str, accepted: bool) -> None:
        cur = self.conn.cursor()
        cur.execute("UPDATE examples SET accepted=? WHERE sha256=?;", (int(accepted), sha))
        self.conn.commit()

    def get_example_id_by_sha(self, sha: str) -> Optional[int]:
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM examples WHERE sha256=?;", (sha,))
        row = cur.fetchone()
        return int(row["id"]) if row else None

    def get_embedding_by_sha(self, sha: str) -> Optional[np.ndarray]:
        cur = self.conn.cursor()
        cur.execute("SELECT embedding, dim FROM examples WHERE sha256=?;", (sha,))
        row = cur.fetchone()
        if not row: return None
        return self._blob_to_np(row["embedding"], int(row["dim"]))

    # ---------- labels ----------
    def add_label(self, sha: str, label: str, kind: str, source: str, score: float = 1.0) -> None:
        assert kind in ("room", "feat")
        assert source in ("user", "model")
        ex_id = self.get_example_id_by_sha(sha)
        if ex_id is None:
            raise ValueError(f"unknown sha: {sha}")
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO labels (example_id, label, kind, source, score, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            (ex_id, label, kind, source, float(score), int(time.time()))
        )
        self.conn.commit()

    # ---------- memory for active learning ----------
    def get_memory(self, kind: str) -> Tuple[np.ndarray, List[set[str]]]:
        """
        accepted=1 olan örneklerin embedding'lerini ve USER etiketlerini döner.
        kind: 'room' | 'feat'
        """
        assert kind in ("room", "feat")
        cur = self.conn.cursor()
        cur.execute("""
            SELECT e.id, e.embedding, e.dim,
                   GROUP_CONCAT(l.label, '||') AS labels
            FROM examples e
            JOIN labels l ON l.example_id = e.id
            WHERE e.accepted=1 AND l.source='user' AND l.kind=?
            GROUP BY e.id
        """, (kind,))
        rows = cur.fetchall()
        if not rows:
            return np.zeros((0, 1), dtype=np.float32), []

        embs, lbl_sets = [], []
        for r in rows:
            embs.append(self._blob_to_np(r["embedding"], int(r["dim"])))
            if r["labels"] is None:
                lbl_sets.append(set())
            else:
                lbl_sets.append(set([s for s in str(r["labels"]).split("||") if s]))
        return np.vstack(embs).astype(np.float32), lbl_sets
