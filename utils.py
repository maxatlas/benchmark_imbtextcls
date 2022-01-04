import os
import datasets
import itertools

import torch

from collections import defaultdict
import matplotlib.pyplot as plt
from datasets import load_dataset
import gensim.downloader as api
from pathlib import Path
from os import rename
from torch.nn.utils.rnn import pad_sequence


def pad_sequence_to_length(text_ids, max_length, batch_first=True, padding_value=0):
    """

    :param text_ids: [torch.tensor]
    :return:
    """
    tensor_of_interest = text_ids[0]
    length = tensor_of_interest.size(0)
    len_to_pad = max_length - length
    tensor_of_interest = torch.cat((tensor_of_interest, torch.tensor([0]*len_to_pad)))
    text_ids[0] = tensor_of_interest

    text_ids = pad_sequence(text_ids, batch_first=batch_first, padding_value=padding_value)
    return text_ids

def preprocess_text(text:str):
    text = text.lower()
    text = text.replace("//", " ")
    text = text.replace("\\", " ")
    text = text.replace("\n", " ")
    text = text.replace("\\n", " ")
    # text = text[1:] if text.startswith(" ") else text
    return text


def get_feature_from_all_splits(ds: datasets.dataset_dict.DatasetDict, texts_field: list, label_field:str):
    def _is_unsupervised_split(ds: datasets.dataset_dict.DatasetDict, key: str, label_field: str):
        """
        return True if labels are -1.
        return False if else and labels are lists.
        """
        res, sample = False, ds[key][label_field][0]

        try:
            return sample < 0
        except TypeError:
            return res

    def _get_data_from_split(ds: datasets.arrow_dataset.Dataset, texts_field: list):
        out = []
        for tf in texts_field:
            if type(ds[0][tf]) is list: data = [preprocess_text(" ".join(ds[i][tf])) for i in range(len(ds))]
            if type(ds[0][tf]) is str: data = [preprocess_text(ds[i][tf]) for i in range(len(ds))]

            if not out: out = data
            # combine with content of the last field.
            else: out = list(map(lambda x, y: x + " " + y, out, data))
        return out

    if type(texts_field) is str: texts_field = [texts_field]

    data, labels = [], []
    for key in ds:
        if not _is_unsupervised_split(ds, key, label_field):
            data_split = _get_data_from_split(ds[key], texts_field)
            labels_split = ds[key][label_field]
            assert len(data_split) == len(labels_split)
            data += data_split
            labels += labels_split

    return data, labels


def get_kv(kvtype:str):
    """

    :param kvtype: "glove"/"word2vec"/"fasttext"
    :return: the weight matrix as KeyVectors
    """
    from config import kvtypes
    kv = api.load(kvtypes.get(kvtype))
    return kv


def count_label(label_list) -> dict:
    d = defaultdict(int)
    for i in label_list:
        if type(i) == list:
            for ix in i: d[ix] += 1
        if type(i) == float:
            i = i>0.5
            d[i] += 1
        if type(i) == int or type(i)==str: d[i] += 1
    return d


def count_word(words: itertools.chain) -> dict:
    d = defaultdict(int)
    for word in words:
        d[word] += 1
    return d


def label_sample(tds) -> dict:
    d = defaultdict(list)
    for label, data in zip(tds.labels, tds.data):
        if type(label) == list:
            for l in label: d[l].append(data)
        if type(label) == int:
            d[label].append(data)
    return d


def display_info(*ds_name, split="train", label_field="label"):
    ds = load_dataset(*ds_name)
    label_list = ds[split][label_field]

    d = count_label(label_list)

    plt.bar(range(len(d.keys())), d.values())
    print(ds)
    plt.show()


def get_imb_factor(no_by_cls:dict, tolerance=0.2):
    imb_classes = 0
    counts = sorted(no_by_cls.values())
    avg = sum(counts) / len(no_by_cls)
    for count in counts:
        if count > avg * (1 + tolerance) or count < avg * (1 - tolerance): imb_classes += 1
    return imb_classes/len(no_by_cls)


def is_imbalanced_ds(no_by_cls:dict, tolerance=0.2, threshold=0.5):
    """
    Imbalanced classes > threshold will be deemed as imbalanced.

    :param tds:
    :param tolerance:
    :param threshold:
    :return:
    """
    imb_factor = get_imb_factor(no_by_cls, tolerance=tolerance)
    if imb_factor > threshold: return True
    return False


def is_imbalanced_cls(cls:str, count_dict:dict, tolerance=0.2):
    avg = sum(count_dict.values())/len(count_dict)
    return count_dict[cls]/avg > (1+tolerance) or count_dict[cls]/avg < (1-tolerance)


def check_imbalance_all(tol, thre, folder="dataset",):
    from TaskDataset import TaskDataset
    paths = [str(x) for x in Path('./%s' % folder).glob('*.tds')]
    tds = TaskDataset()
    imb_dataset = 0
    for path in paths:
        tds.load(path)
        imb_factor = get_imb_factor(tds, tolerance=tol)
        print("%s: %f" % (path, imb_factor))
        if imb_factor > thre: imb_dataset += 1

    print("%i datasets out of %i datasets are imbalanced." %(imb_dataset, len(paths)))


def rename_files(folder, old_str, new_str):
    paths = [str(f) for f in Path('./%s' % folder).glob("*%s" %old_str)]
    for p in paths:
        rename(p, p.replace(old_str, new_str))
    print(os.listdir(folder))


# if __name__ == "__main__":
#     check_imbalance_all(0.3, 0.6, "dataset")



