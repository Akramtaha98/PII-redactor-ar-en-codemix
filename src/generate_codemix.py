import json, random, argparse
from datasets import load_dataset

EN_NAMES = ["John","Sara","Michael","Leila","Omar","Nour","Ahmed","Mona","Hassan","Khalid"]
AR_NAMES = ["ليلى","أحمد","خالد","منى","نور","سارة","محمد","علي","سلمان","حسين"]
CONNECTORS = ["and","about","regarding","re:","re","re."]
AR_CONNECTORS = ["و","عن","بخصوص"]

def make_mixed_sentence(en_tokens, ar_tokens):
    en = " ".join(en_tokens[:10])
    ar = " ".join(ar_tokens[:10])
    return f"{ar} {random.choice(AR_CONNECTORS)} {random.choice(EN_NAMES)} {random.choice(CONNECTORS)} {en} {random.choice(AR_NAMES)}"

def main(out_path, n):
    en_ds = load_dataset("wikiann", "en")["train"]
    ar_ds = load_dataset("wikiann", "ar")["train"]
    with open(out_path, "w", encoding="utf-8") as out_f:
        for _ in range(n):
            en_ex = en_ds[random.randrange(len(en_ds))]["tokens"]
            ar_ex = ar_ds[random.randrange(len(ar_ds))]["tokens"]
            text = make_mixed_sentence(en_ex, ar_ex)
            out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"Wrote {n} mixed sentences to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    main(args.out, args.n)