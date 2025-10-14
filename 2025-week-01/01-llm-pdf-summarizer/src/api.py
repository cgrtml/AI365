from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from src.utils import extract_text_from_pdf, summarize

load_dotenv()
app = FastAPI(title="AI365 Summarizer API")

@app.post("/summarize")
async def summarize_pdf(
    pdf: UploadFile = File(...),
    method: str = Form("textrank"),  # "openai" veya "textrank"
    sentences: int = Form(10),
):
    content = await pdf.read()
    tmp_path = f"/tmp/{pdf.filename}"
    with open(tmp_path, "wb") as f:
        f.write(content)

    text = extract_text_from_pdf(tmp_path)
    if not text.strip():
        return JSONResponse({"error": "No extractable text (needs OCR)."}, status_code=400)

    summary = summarize(text, method=method, sentences=sentences)
    return {"method": method, "sentences": sentences, "summary": summary}
