import spacy
import itertools

from time import time
from dill import dump, load
from TaskDataset import TaskDataset, split_tds
from datasets import load_dataset, dataset_dict, ClassLabel
from config import *
from pathlib import Path
from torchtext.data import get_tokenizer
from utils import (get_kv, get_feature_from_all_splits,
                   count_word)

"""

    Takes [(text, label)]
    tokenize, embed -> id, decided by the model type
    split into train, test, validation by split_config
    Gives {(id list or dict of lists, label_id)}

"""


def get_labels_meta(label_list):
    labels = set()
    if type(label_list[0]) is not list:
        for l in label_list: labels.add(l)
    else:
        for ls in label_list:
            for l in ls: labels.add(l)
    labels = sorted(list(labels))
    labels_meta = ClassLabel(names=labels)
    return labels_meta


def replace_tkndata_to_index(filepath:str, w2ipath:str):
    tds = TaskDataset()
    tds.load(filepath)

    save_file = filepath.replace(".tkn", ".wi")

    w2i = load(open(w2ipath, "rb"))
    data = tds.data
    data = [[w2i.get(word, 0) for word in doc] for doc in data]

    tds.data = data
    tds.save(save_file)

    return tds


def replace_all_tkndata_to_index(w2ipath:str, folder_path="dataset"):
    tds = TaskDataset()
    file_paths = [str(file) for file in Path("./%s" %folder_path).glob('*.tkn')]
    for path in file_paths:
        tds.load(path)
        replace_tkndata_to_index(tds, w2ipath)


def build_emb_layer_from_all_datasets(dataset_folder="dataset", cutoff=400_000):
    import itertools
    import torch

    from TaskDataset import TaskDataset
    from Classifier import build_emb_layer

    file_paths = [str(file) for file in Path("./%s" %dataset_folder).glob('*.tkn')]
    tds = TaskDataset()

    tkndata = iter(itertools.chain.from_iterable(tds.load(path).data for path in file_paths))
    tkndata = iter(itertools.chain.from_iterable(tkndata))
    print("Counting words ...")

    start = time()
    count_dict = count_word(tkndata)
    end = time()
    print("Done. Takes %f seconds." %(end-start))

    del count_dict[' ']
    dump(count_dict, open("models/word_count_full", "wb"))

    tkndata = [(key, count_dict[key]) for key in count_dict]
    tkndata = sorted(tkndata, key=lambda x:x[1], reverse=True)[:cutoff]
    tkndata = [key for (key, count) in tkndata]
    tkndata = ["[PAD]"] + tkndata

    word2index = {word: i for i, word in enumerate(tkndata)}

    print("word2index length: %i" %len(word2index))
    dump(word2index, open("models/word_index", "wb"))

    print("vocab size: %i" %len(tkndata))

    for kvtype in kvtypes:
        print("building embedding layer for %s ..." %kvtype)
        emb_layer, unfound_words = build_emb_layer(tkndata, get_kv(kvtype))
        dump(unfound_words, open("models/unfound_words_%s"%kvtype, "wb"))
        torch.save(emb_layer.state_dict(), "models/emb_layer_%s" % kvtype)
        print("Done.")


    print("Done for all kvtypes")


def prerun_per_dataset(dconfig, tokenizer, save_folder="dataset", suffix=""):
    """
    Handles Huggingface dataset -> word_id

    :param dconfig:
    :param save_folder:
    :param suffix:
    :return:
    """
    hds = load_dataset(*dconfig['dname'])
    # Merge relevant text fields
    texts, labels = get_feature_from_all_splits(hds, texts_field=dconfig['text_fields'],
                                                label_field=dconfig['label_field'])
    # set binary labels if labels are float
    if type(labels[0]) == float:
        labels = [0 if label < 0.5 else 1 for label in labels]

    assert len(labels) == len(texts)

    # Use Huggingface's ClassLabel if exists else create one with labels.
    labels_meta = hds['train'].features[dconfig['label_field']] if \
        type(hds['train'].features[dconfig['label_field']]) is ClassLabel else \
        get_labels_meta(labels)

    path = "%s/%s" % (save_folder, "_".join(dconfig['dname']))+suffix
    print(path)
    tds = TaskDataset(dname=dconfig['dname'], labels=labels, data=texts,
                      labels_meta=labels_meta)
    tds.save(path+".tds", suffix=suffix)

    if dconfig['label_field'] == "product_category":
        docs = tds.load("dataset/amazon_reviews_multi_en_stars.tkn").data
    else:
        if tokenizer == "spacy":
            nlp = spacy.load("en_core_web_sm")
            docs = nlp.pipe(texts, n_process=2, batch_size=2000)
            docs = [[tok.text for tok in doc] for doc in docs]
        else:
            tokenizer = get_tokenizer("basic_english")
            docs = [tokenizer(text) for text in texts]

    tds = TaskDataset(dname=dconfig['dname'], labels=labels, data=docs,
                      labels_meta=labels_meta)
    tds.save(path+".tkn", suffix=suffix)


def main(folder):
    for dmeta in datasets_meta[:0]:
        print("Pretrain %s ..." % dmeta['dname'])
        suffix = ""
        if dmeta['dname'] == ["amazon_reviews_multi", "en"]: suffix = "_%s" % dmeta['label_field']

        start = time()
        tokenizer = "not spacy" if dmeta['dname'][0].startswith("lex") else "spacy"
        prerun_per_dataset(dmeta, tokenizer=tokenizer, suffix=suffix)
        end = time()
        print("costs %f seconds." % (end - start))

    print("Building embedding layer ...")
    build_emb_layer_from_all_datasets(folder, training_config['cutoff'])

    print(".tkn -> .wi")
    file_paths = [str(file) for file in Path("./%s" % folder).glob('*.tkn')]
    for file_path in file_paths[:]:
        print("\n"+file_path)
        print("Transforming tokenized text to word index ...")
        replace_tkndata_to_index(file_path, w2ipath="models/word_index")
        file_path = file_path.replace(".tkn", ".wi")
    print("Done")


if __name__ == "__main__":
    main("dataset")
