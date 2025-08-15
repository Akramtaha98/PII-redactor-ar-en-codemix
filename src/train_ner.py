import os, argparse, json
from typing import List
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np

LABEL_LIST = ["O","B-LOC","I-LOC","B-PER","I-PER","B-ORG","I-ORG"]
LABEL_TO_ID = {l:i for i,l in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i:l for l,i in LABEL_TO_ID.items()}

def load_wikiann_langs(langs: List[str]) -> DatasetDict:
    parts = [load_dataset("wikiann", lc) for lc in langs]
    def cat(split):
        ds = [p[split] for p in parts]
        return concatenate_datasets(ds) if len(ds) > 1 else ds[0]
    return DatasetDict(train=cat("train"), validation=cat("validation"), test=cat("test"))

def load_codemix_jsonl(path: str):
    from datasets import Dataset
    data = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
    toks = [rec["text"].split() for rec in data]
    tags = [["O"]*len(t) for t in toks]
    return Dataset.from_dict({"tokens": toks, "ner_tags": [[LABEL_TO_ID[t] for t in ts] for ts in tags]})

def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=False):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous = None
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != previous:
                label_ids.append(label[wid])
            else:
                label_ids.append(label[wid] if label_all_tokens else -100)
            previous = wid
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main(args):
    ds = load_wikiann_langs(args.langs)
    if args.codemix and os.path.exists(args.codemix):
        cm = load_codemix_jsonl(args.codemix)
        ds["train"] = concatenate_datasets([ds["train"], cm])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized = ds.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=len(LABEL_LIST), id2label=ID_TO_LABEL, label2id=LABEL_TO_ID)

    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["query","key","value","dense"]
    )
    model = get_peft_model(model, peft_config)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=float(args.lr),
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50
    )

    from evaluate import load as load_metric
    seqeval = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_labels = [[ID_TO_LABEL[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [ID_TO_LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"]}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--output_dir", default="outputs/ner_lora")
    ap.add_argument("--langs", nargs="+", default=["ar","en"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", default="2e-5")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--codemix", default=None)
    args = ap.parse_args()
    main(args)