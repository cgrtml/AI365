## CLI
python -m src.main --pdf notebooks/sample.pdf --method textrank --sentences 8
# OpenAI:
python -m src.main --pdf notebooks/sample.pdf --method openai

## API
uvicorn src.api:app --reload
# Open http://127.0.0.1:8000/docs and POST /summarize
