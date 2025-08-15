import json, argparse
from .redact import redact_text

def spans_iou(a, b):
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0

def evaluate(model_dir: str, labeled_jsonl: str, iou_threshold: float = 0.5):
    gold = [json.loads(l) for l in open(labeled_jsonl, "r", encoding="utf-8")]
    tp = fp = fn = 0
    for rec in gold:
        text = rec["text"]
        gspans = rec["entities"]
        redacted, meta = redact_text(text, model_dir)
        pspans = meta["merged_spans"]
        matched = set()
        for p in pspans:
            found = False
            for i, g in enumerate(gspans):
                if i in matched: continue
                if spans_iou({"start":p["start"],"end":p["end"]}, g) >= iou_threshold:
                    tp += 1; matched.add(i); found = True; break
            if not found: fp += 1
        fn += len(gspans) - len(matched)
    precision = tp / (tp + fp) if (tp+fp)>0 else 0.0
    recall = tp / (tp + fn) if (tp+fn)>0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    print(json.dumps({"precision": precision, "recall": recall, "f1": f1}, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--labeled_jsonl", required=True)
    ap.add_argument("--iou_threshold", type=float, default=0.5)
    args = ap.parse_args()
    evaluate(args.model_dir, args.labeled_jsonl, args.iou_threshold)