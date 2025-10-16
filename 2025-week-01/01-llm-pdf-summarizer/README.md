# 📄 LLM PDF Summarizer

LLM PDF Summarizer is a lightweight **CLI & API tool** that automatically summarizes text-based PDF files using either:
- 🧠 **TextRank** (offline, open-source summarization)
- 🤖 **OpenAI GPT models** (for more natural, context-aware summaries)

It’s ideal for developers, researchers, students, analysts, and anyone who works with large PDFs and wants quick, structured summaries without manually reading everything.

---

## 🚀 Usage

### CLI
```bash
python -m src.main --pdf notebooks/sample.pdf --method textrank --sentences 8
# OpenAI:
python -m src.main --pdf notebooks/sample.pdf --method openai

API
uvicorn src.api:app --reload
# Open http://127.0.0.1:8000/docs and POST /summarize

Docker
docker run --rm -p 8000:8000 cagritemel/ai365-pdf:latest
# Then open http://127.0.0.1:8000/docs

👥 Who Can Use It
🧑‍💻 Developers & Data Scientists — integrate summarization into pipelines or apps
🧑‍🏫 Students & Academics — summarize research papers in seconds
📝 Journalists & Analysts — distill legal and policy PDFs quickly
🧠 Open-source contributors — extend with new summarization methods

📜 License
MIT License — free to use, modify, and contribute.

