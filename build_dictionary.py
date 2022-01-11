import dill
import spacy
import numpy

from time import time
from collections import defaultdict
from vars import datasets_meta, parameter_folder
from nltk.tokenize import word_tokenize
from datasets import load_dataset

from dataset_utils import preprocess_texts, preprocess_text


def nltk_run():
    tokens_count = defaultdict(int)
    for i, meta in enumerate(datasets_meta):
        if meta['huggingface_dataset_name'] == ["lex_glue", "ecthr_b"]:
            continue

        data_name, label_field, text_fields = meta.values()
        print(i, data_name)
        ds = load_dataset(*data_name)
        for key in ds:
            print("\t" + key)
            df = ds[key].to_pandas()
            for field in text_fields:
                print("\t\t" + field)
                docs = list(df[field])
                for doc in docs:
                    if type(doc) is numpy.ndarray:
                        doc = "\n".join(doc)
                    doc = preprocess_text(doc)
                    tokens = word_tokenize(doc)
                    for token in tokens:
                        tokens_count[token] += 1

        print(len(tokens_count))
        dill.dump(tokens_count, open("%s/word_count_nltk" % parameter_folder, "wb"))


def spacy_run():
    nlp = spacy.load("en_core_web_sm")
    tokens_count = defaultdict(int)
    # tokens_count = dill.load(open("parameters/word_count_spacy", "rb"))
    for i, meta in enumerate(datasets_meta[:]):
        if meta['dname'] == ["lex_glue", "ecthr_b"]:
            continue

        clock_i = time()
        data_name, label_field, text_fields = meta.values()
        print(i, data_name)
        ds = load_dataset(*data_name)
        for key in ds:
            print("\t" + key)
            df = ds[key].to_pandas()
            for field in text_fields:
                print("\t\t" + field)
                docs = list(df[field])
                if type(docs[0]) is str:
                    docs = preprocess_texts(docs)
                    docs = nlp.pipe(docs, n_process=2, disable=["tok2vec", "transformer"])
                    tokens = [[tok.text for tok in doc] for doc in docs]
                    for doc in tokens:
                        for token in doc:
                            tokens_count[token] += 1
                else:
                    for sub_docs in docs:
                        sub_docs = preprocess_texts(sub_docs)
                        sub_docs = nlp.pipe(sub_docs, n_process=4, disable=["tok2vec", "transformer"])
                        tokens = [[tok.text for tok in doc] for doc in sub_docs]
                        for doc in tokens:
                            for token in doc:
                                tokens_count[token] += 1

        print("%i words recorded so far." % len(tokens_count))
        print("%f seconds." % (time() - clock_i))
        dill.dump(tokens_count, open("%s/word_count_spacy" % parameter_folder, "wb"))


if __name__ == "__main__":
    spacy_run()
    nltk_run()