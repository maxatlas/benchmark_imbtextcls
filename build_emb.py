import dill
import numpy

from collections import defaultdict
from task_config import datasets_meta
from nltk.tokenize import word_tokenize
from datasets import load_dataset

from utils import preprocess_text

if __name__ == "__main__":
    tokens_count = defaultdict(int)
    for i, meta in enumerate(datasets_meta):
        data_name, label_field, text_fields = meta.values()
        print(i, data_name)
        ds = load_dataset(*data_name)
        for key in ds:
            print("\t"+key)
            df = ds[key].to_pandas()
            for field in text_fields:
                print("\t\t"+field)
                docs = list(df[field])
                for doc in docs:
                    if type(doc) is numpy.ndarray:
                        doc = "\n".join(doc)
                    doc = preprocess_text(doc)
                    tokens = word_tokenize(doc)
                    for token in tokens:
                        tokens_count[token] += 1

        print(len(tokens_count))
        dill.dump(tokens_count, open("parameters/word_count_nltk", "wb"))
