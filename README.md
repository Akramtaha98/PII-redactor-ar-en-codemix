# PII Redactor for Arabic–English Code-Mixed Text

A lightweight hybrid system that **redacts PII** (names, orgs, locations, email, phone, URLs, IPs) from Arabic–English code-mixed text.

- Multilingual NER fine-tuned with **LoRA** (parameter-efficient).
- **Regex** rules for structured PII.
- CPU-friendly inference, simple CLI.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train the NER with LoRA (stage 1 + stage 2)

Stage 1: Arabic + English (WikiANN).  
Stage 2: add synthetic code-mix augmentation.

```bash
python src/train_ner.py --model xlm-roberta-base --output_dir outputs/ner_lora   --langs ar en --epochs 3 --batch_size 16

python src/generate_codemix.py --out data/codemix.jsonl --n 3000

python src/train_ner.py --model xlm-roberta-base --output_dir outputs/ner_lora_codemix   --langs ar en --epochs 1 --batch_size 16 --resume_from outputs/ner_lora   --codemix data/codemix.jsonl
```

> If you have **ANERcorp**, place it at `data/anercorp/` (train/dev/test as CONLL-like files). The trainer will auto-detect and include it.

### Run the redactor on a text file

```bash
python src/infer_redact.py --model_dir outputs/ner_lora_codemix   --in_file examples/sample.txt --out_file outputs/redacted.txt
```

### Evaluate PII redaction on a small labeled set (optional)

```bash
python src/eval_pii.py --model_dir outputs/ner_lora_codemix   --labeled_jsonl examples/labeled_eval.jsonl
```

## Project Structure

```
src/
  data_utils.py          # load WikiANN, optional ANERcorp, tokenize, label maps
  generate_codemix.py    # make synthetic Ar–En code-mixed sentences
  train_ner.py           # LoRA fine-tuning for token classification
  rules.py               # regex + phone rules for structured PII
  redact.py              # redaction core (NER + rules fusion)
  infer_redact.py        # CLI for redaction
  eval_pii.py            # evaluate precision/recall for PII redaction
  utils.py               # misc helpers (digit normalization, etc.)
examples/
  sample.txt             # example code-mixed text
  labeled_eval.jsonl     # tiny demo set with PII annotations
configs/
  train_config.yaml      # example config
```

## Notes

- **Datasets**: `datasets` will download WikiANN (`wikiann`, configs `ar`, `en`). ANERcorp is optional (place locally).
- **Labels**: This project uses PER/ORG/LOC. You can extend with more PII labels if you provide training data.
- **Simulated expectations**: The paper’s exact numbers are simulated. Run your own training to produce real metrics.
- **License**: MIT.
