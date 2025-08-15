import re
import phonenumbers
from typing import List, Dict
from .utils import normalize_arabic_digits

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
GENERIC_NUM_RE = re.compile(r"(?<!\d)\d{7,15}(?!\d)")

def find_structured_pii(text: str, default_region: str = "IQ") -> List[Dict]:
    spans = []
    t = normalize_arabic_digits(text)

    for m in EMAIL_RE.finditer(t):
        spans.append({"start": m.start(), "end": m.end(), "label": "EMAIL"})
    for m in URL_RE.finditer(t):
        spans.append({"start": m.start(), "end": m.end(), "label": "URL"})
    for m in IPV4_RE.finditer(t):
        spans.append({"start": m.start(), "end": m.end(), "label": "IP"})

    for m in GENERIC_NUM_RE.finditer(t):
        num = t[m.start():m.end()]
        try:
            parsed = phonenumbers.parse(num, default_region)
            if phonenumbers.is_valid_number(parsed):
                spans.append({"start": m.start(), "end": m.end(), "label": "PHONE"})
        except Exception:
            pass

    spans = sorted(spans, key=lambda x: (x["start"], -(x["end"]-x["start"])))
    merged = []
    for s in spans:
        if not merged or s["start"] >= merged[-1]["end"]:
            merged.append(s)
        else:
            prev = merged[-1]
            if (s["end"] - s["start"]) > (prev["end"] - prev["start"]):
                merged[-1] = s
    return merged