import argparse, json
from .redact import redact_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    text = open(args.in_file, "r", encoding="utf-8").read()
    redacted, meta = redact_text(text, args.model_dir, threshold=args.threshold)
    open(args.out_file, "w", encoding="utf-8").write(redacted + "\n")
    open(args.out_file + ".meta.json", "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved redacted text to {args.out_file}")
    print(f"Saved spans meta to {args.out_file}.meta.json")

if __name__ == "__main__":
    main()