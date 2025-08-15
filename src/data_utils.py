from typing import List
from datasets import load_dataset, DatasetDict, concatenate_datasets

def load_wikiann_langs(lang_codes: List[str]) -> DatasetDict:
    parts = [load_dataset("wikiann", lc) for lc in lang_codes]
    def cat(split):
        ds = [p[split] for p in parts]
        return concatenate_datasets(ds) if len(ds) > 1 else ds[0]
    return DatasetDict(train=cat("train"), validation=cat("validation"), test=cat("test"))