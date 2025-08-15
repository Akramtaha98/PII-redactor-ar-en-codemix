from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from .rules import find_structured_pii

LABEL_LIST = ["O","B-LOC","I-LOC","B-PER","I-PER","B-ORG","I-ORG"]
ID_TO_LABEL = {i:l for i,l in enumerate(LABEL_LIST)}

def ner_spans(text: str, tokenizer, model, threshold: float = 0.5):
    tokens = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model(**tokens)
        probs = out.logits.softmax(-1)[0]
    offsets = tokens["offset_mapping"][0].tolist()
    labels = probs.argmax(-1).tolist()
    spans = []
    current = None
    for i, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            continue
        label_id = labels[i]
        label = ID_TO_LABEL[label_id]
        score = probs[i][label_id].item()
        if label == "O" or score < threshold:
            if current:
                spans.append(current); current = None
            continue
        etype = label.split("-")[-1]
        if label.startswith("B-"):
            if current: spans.append(current)
            current = {"start": start, "end": end, "label": etype}
        elif label.startswith("I-") and current and current["label"] == etype:
            current["end"] = end
        else:
            if current: spans.append(current)
            current = None
    if current:
        spans.append(current)
    return spans

def redact_text(text: str, model_dir: str, threshold: float = 0.5):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    nspans = ner_spans(text, tokenizer, model, threshold=threshold)
    rspans = find_structured_pii(text)

    all_spans = sorted(nspans + rspans, key=lambda x: x["start"])
    merged = []
    for s in all_spans:
        if not merged or s["start"] > merged[-1]["end"]:
            merged.append(s.copy())
        else:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
            if merged[-1]["label"] != s["label"]:
                merged[-1]["label"] = "PII"

    out = []
    last = 0
    for s in merged:
        out.append(text[last:s["start"]])
        tag = s["label"]
        rep = f"[{tag}]" if tag in {"PER","ORG","LOC","EMAIL","PHONE","URL","IP"} else "[PII]"
        out.append(rep)
        last = s["end"]
    out.append(text[last:])
    return "".join(out), {"ner_spans": nspans, "rule_spans": rspans, "merged_spans": merged}