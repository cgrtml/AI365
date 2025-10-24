from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from src.utils import caption_image_from_bytes

load_dotenv()
app = FastAPI(title="AI365 Image Captioner")

@app.post("/caption")
async def caption_endpoint(
    img: UploadFile = File(...),
    method: str = Form("offline")  # "offline" | "openai"
):
    content = await img.read()
    try:
        text = caption_image_from_bytes(content, method=method)
        return {"method": method, "caption": text}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
