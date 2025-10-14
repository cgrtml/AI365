import os
from typing import List
from pypdf import PdfReader
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk

# OpenAI SDK (varsa kullanacağız; yoksa TextRank'e düşer)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def ensure_nltk():
    # yeni NLTK sürümlerinde 'punkt' + 'punkt_tab' gerekiyor
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def summarize_textrank(text: str, sentence_count: int = 10) -> str:
    ensure_nltk()
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    return "\n".join(str(s) for s in summary_sentences)


def _chunk(text: str, max_chars: int = 2500) -> List[str]:
    chunks, buf, size = [], [], 0
    for line in text.splitlines():
        if size + len(line) + 1 > max_chars:
            chunks.append("\n".join(buf))
            buf, size = [line], len(line) + 1
        else:
            buf.append(line)
            size += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return [c for c in chunks if c.strip()]


def summarize_openai(text: str, model: str = "gpt-4o-mini") -> str:
    """OpenAI ile özet; API anahtarı yoksa otomatik TextRank'e düşer."""
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return summarize_textrank(text, sentence_count=10)

    client = OpenAI()  # anahtarı ortamdan okur
    parts = _chunk(text, max_chars=2500)

    bullets = []
    for ch in parts:
        r = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a concise scientific summarizer."},
                {"role": "user", "content": f"Summarize into 6-8 bullet points:\n\n{ch}"}
            ],
        )
        bullets.append(r.choices[0].message.content.strip())

    merged = "\n".join(bullets)
    final = client.chat.completions.create(
        model=model, temperature=0.2,
        messages=[
            {"role": "system", "content": "Produce a single, cohesive summary."},
            {"role": "user", "content": f"Combine the bullets below into ~10 bullets + a short paragraph:\n\n{merged}"}
        ],
    )
    return final.choices[0].message.content.strip()


def summarize(text: str, method: str = "textrank", sentences: int = 10) -> str:
    """API ve CLI tarafından çağrılan ortak fonksiyon."""
    m = (method or "textrank").lower()
    if m == "openai":
        return summarize_openai(text)
    return summarize_textrank(text, sentence_count=sentences)
