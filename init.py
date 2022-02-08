import dill
import spacy
import numpy as np
import build_model
import vars
from time import time
from collections import defaultdict
from vars import datasets_meta, parameter_folder
from nltk.tokenize import word_tokenize
from datasets import load_dataset

from dataset_utils import preprocess_texts, preprocess_text

from vars import cutoff, kvtypes

import gensim.downloader as api

# api.BASE_DIR = vars.hpc_folder+".cache/gensim-data/"
# api.base_dir = vars.hpc_folder+".cache/gensim-data/"
# api._create_base_dir()

# os.environ['TRANSFORMERS_CACHE'] = vars.hf_cache_folder+"/modules"
# os.environ['HF_DATASETS_CACHE'] = vars.hf_cache_folder+"/datasets"


import nltk
nltk.download("punkt")


def cache_needed_data():
    print(api.base_dir)
    for meta in datasets_meta:
        load_dataset(*meta['huggingface_dataset_name'])
    for name in kvtypes.values():
        api.load(name)


def save_all_transformer_emb_layer():
    from Config import ModelConfig
    from model_utils import save_transformer_emb
    names = [("bert", "bert-base-uncased"),
             ("xlnet", "xlnet-base-cased"),
             ("roberta", "roberta-base")]
    for model_name, pretrained_name in names[-1:]:
        print("\n"+model_name)
        mc = ModelConfig(model_name, 2, pretrained_model_name=pretrained_name)
        model = build_model.main(mc)
        save_transformer_emb(model, model_name)
        print("Done.")


def build_emb_layers(count_dict_path):
    import torch
    import torch.nn as nn
    from gensim.models.keyedvectors import KeyedVectors

    def get_kv(kvtype: str):
        """
        :param kvtype: "glove"/"word2vec"/"fasttext"
        :return: the weight matrix as KeyVectors
        """
        from vars import kvtypes
        kv = api.load(kvtypes.get(kvtype))
        return kv

    count_dict = dill.load(open(count_dict_path, "rb"))

    def build_emb_layer(tknwords: list, kv: KeyedVectors, trainable=1):
        def _create_weight_matrix(start_i):
            wm = np.zeros((num_emb, emb_dim))
            for i, word in enumerate(tknwords[start_i:]):
                try:
                    wm[i + start_i] = kv[word]
                except KeyError:
                    wm[i + start_i] = np.random.normal(scale=0.6, size=(emb_dim,))
                    unfound_words.add(word)
            wm = torch.tensor(wm)
            return wm

        unfound_words = set()
        num_emb, emb_dim = len(tknwords), len(kv['the'])
        word_start_i = 2
        emb_layer = nn.Embedding(num_emb, emb_dim)
        emb_layer.load_state_dict({'weight': _create_weight_matrix(word_start_i)})
        if not trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, unfound_words

    tkndata = [(key, count_dict[key]) for key in count_dict]
    tkndata = sorted(tkndata, key=lambda x: x[1], reverse=True)[:cutoff]
    tkndata = [key for (key, count) in tkndata]
    tkndata = ["[PAD]", "[UNK]"] + tkndata

    word2index = {word: i for i, word in enumerate(tkndata)}

    print("word2index length: %i" % len(word2index))
    dill.dump(word2index, open("%s/word_index" % parameter_folder, "wb"))

    print("vocab size: %i" % len(tkndata))

    for kvtype in kvtypes:
        print("building embedding layer for %s ..." % kvtype)
        emb_layer, unfound_words = build_emb_layer(tkndata, get_kv(kvtype))
        print("emb layer size: %s" % str(emb_layer.weight.shape))
        dill.dump(unfound_words, open("%s/unfound_words_%s" % (parameter_folder, kvtype), "wb"))
        torch.save(emb_layer.state_dict(), "%s/emb_layer_%s" % (parameter_folder, kvtype))
        print("Done.")

    print("Done for all kvtypes")


def nltk_run():
    tokens_count = defaultdict(int)
    for i, meta in enumerate(datasets_meta):
        if meta['huggingface_dataset_name'] == ["lex_glue", "ecthr_b"]:
            continue

        data_name, label_field, text_fields, _, _ = meta.values()
        print(i, data_name)
        ds = load_dataset(*data_name)
        for key in ds:
            print("\t" + key)
            df = ds[key].to_pandas()
            for field in text_fields:
                print("\t\t" + field)
                docs = list(df[field])
                for doc in docs:
                    if type(doc) is np.ndarray:
                        doc = "\n".join(doc)
                    doc = preprocess_text(doc)
                    tokens = word_tokenize(doc)
                    for token in tokens:
                        tokens_count[token] += 1

        print(len(tokens_count))
        dill.dump(tokens_count, open("%s/word_count_nltk" % parameter_folder, "wb"))


def spacy_run():
    nlp = spacy.load("en_core_web_sm")
    # tokens_count = defaultdict(int)
    tokens_count = dill.load(open("%s/word_count_spacy" % vars.parameter_folder, "rb"))
    for i, meta in enumerate(datasets_meta[16:]):
        if meta['huggingface_dataset_name'] == ["lex_glue", "ecthr_b"]:
            continue

        clock_i = time()
        data_name, label_field, text_fields, _, _ = meta.values()
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
    cache_needed_data()
    spacy_run()
    # nltk_run()
    # build_emb_layers("%s/word_count_nltk" % parameter_folder)
    build_emb_layers("%s/word_count_spacy" % parameter_folder)
    save_all_transformer_emb_layer()
