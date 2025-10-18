# server/app.py  — EN ÜSTE KOY
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware

# (Diğer 3rd party importlar)
from typing import List, Dict, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch, io, yaml, os, hashlib

# Yerel modüller (bunlar app oluşturulduktan SONRA değil; ama app tanımı ÜSTTE olmalı)
from store import Store
from active import Active

# ---- FastAPI instance'ını OLUŞTUR ----
app = FastAPI(title="Real Estate Photo Tagger", version="0.2.0")

# ---- CORS MIDDLEWARE'İ EKLE ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
from fastapi import FastAPI  # üstlerde zaten var
# ...

@app.get("/")
def root():
    return {"ok": True}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

