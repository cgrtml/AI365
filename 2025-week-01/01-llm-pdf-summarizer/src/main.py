import argparse
from src.utils import extract_text_from_pdf, summarize_text

def main():
    parser = argparse.ArgumentParser(description="PDF summarizer (TextRank)")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--sentences", type=int, default=10, help="Number of sentences in summary")
    args = parser.parse_args()

    text = extract_text_from_pdf(args.pdf)
    if not text.strip():
        raise SystemExit("No extractable text found. If the PDF is scanned, run OCR first.")
    summary = summarize_text(text, args.sentences)

    print("\n--- 📄 SUMMARY ---\n")
    print(summary)

if __name__ == "__main__":
    main()
