import argparse
from src.utils import caption_image

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to image")
    p.add_argument("--method", choices=["offline","openai"], default="offline")
    args = p.parse_args()

    text = caption_image(args.image, method=args.method)
    print("\n--- 🖼️ CAPTION ---\n")
    print(text)

if __name__ == "__main__":
    main()
