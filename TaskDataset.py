import random
import torch

from config import split_ratio, split_strategies, imb_ratio
from dill import dump, load
from torch.utils.data import Dataset

from copy import deepcopy
from math import floor
from utils import is_imbalanced_ds, is_imbalanced_cls, get_imb_factor

from utils import count_label, label_sample


def format_name(name_list):
    return [name.replace("/", "_") for name in name_list]


class TaskDataset(Dataset):
    def __init__(self, dname=[], data=[], labels=[], attention_mask=[], token_type_ids=[], labels_meta=None):
        """

        :param dname: available options see Dataset Overview sheet column A.
        """
        self.name = format_name(dname)
        self.data = data
        self.labels = labels
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels_meta = labels_meta

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.data[i], self.attention_mask[i] if self.attention_mask else [], self.token_type_ids[i] if self.token_type_ids else [], self.labels[i]

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.data

    def save(self, path="", suffix=""):
        assert type(path) == str

        f = open(path, 'wb')
        d = {
            "name": self.name,
            "labels_meta": self.labels_meta,
            "labels": self.labels,
            "data": self.data,
        }
        dump(d, f)

    def load(self, path=""):
        f = open(path, "rb")
        d = load(f)
        self.name = d['name']
        self.labels_meta = d['labels_meta']
        self.labels = d['labels']
        self.data = d.get('data')

        return self

    def convert_labels_to_list(self):
        from copy import deepcopy
        label_list = deepcopy(self.labels)
        if type(self.labels[0]) is str:
            label_list = [self.labels_meta.names.index(l) for l in self.labels]
        return label_list

    def get_label_ids(self):
        label_ids = [len(self.labels_meta.names) * [0]] * self.__len__()
        label_list = self.convert_labels_to_list()
        for i, labels in enumerate(label_list):
            if type(labels) is int: labels = [labels]
            for label in labels: label_ids[i][label] = 1
        return label_ids

    def set_label_ids(self):
        self.labels = self.get_label_ids()

    def cut_samples(self, limit):
        if type(self.data[0]) is list:
            data = [sample[:limit] for sample in self.data]
            self.data = data
        if self.attention_mask:
            mask = [sample[:limit] for sample in self.attention_mask]
            self.attention_mask = mask


def set_imbalance_by_cls(no_by_cls:dict, tolerance, threshold, a=0.75, b=2.1):
    # choose a ratio of balanced classes, reduce amount. assert is_imbalanced_ds
    # recursively reduce until imbalanced
    out = deepcopy(no_by_cls)
    imb_no = floor(len(no_by_cls) * a)
    imb_no = 1 if not imb_no else imb_no
    cls_count = sorted([(cls, no_by_cls[cls]) for cls in no_by_cls], key=lambda x:x[1], reverse=True)
    bcls_count = [None if is_imbalanced_cls(cls, no_by_cls, tolerance) else (cls, count) for cls, count in cls_count]
    bcls_count = list(filter(lambda x:x != None, bcls_count))[-imb_no:]
    bcls_count = [(cls, floor(count * b)) for cls, count in bcls_count]
    for cls, count in bcls_count:
        out[cls] = count

    print("Split result for training set:")
    print(out)

    print("Checking if the new split is imbalanced ...")
    assert is_imbalanced_ds(out, tolerance, threshold)
    print("Checked.")

    return out


def split_tds(filename:str, split_strategy="uniform"):
    save_file = filename.split("/")[-1].split(".")[0]

    tds = TaskDataset()
    tds.load(filename)

    tolerance, threshold = imb_ratio['tolerance'], imb_ratio['threshold']
    # get sample amount for each split by strategy
    count_dict = count_label(tds.labels)

    test_no_by_label = _get_split_amount(count_dict, split_strategies.index(split_strategy), split_ratio['test'], tolerance)
    train_no_by_label = {lb: floor((count_dict[lb] - test_no_by_label[lb]) * split_ratio['train']) for lb in count_dict}
    val_no_by_label = {lb: count_dict[lb] - test_no_by_label[lb] - train_no_by_label[lb] for lb in count_dict}

    print("Train/Test/Val split:")
    print(train_no_by_label)
    print(test_no_by_label)
    print(val_no_by_label)

    if not is_imbalanced_ds(count_dict):
        a, b = imb_ratio[save_file]
        train_no_by_label = set_imbalance_by_cls(train_no_by_label, tolerance, threshold, a, b)
        assert is_imbalanced_ds(train_no_by_label, tolerance, threshold) or is_imbalanced_ds(train_no_by_label, tolerance-0.1, threshold-0.1)

    # retrieve the sample
    data_dict = label_sample(tds)

    data_dict, test_data, test_labels = _retrieve_sample(data_dict, test_no_by_label)
    data_dict, train_data, train_labels = _retrieve_sample(data_dict, train_no_by_label)
    data_dict, val_data, val_labels = _retrieve_sample(data_dict, val_no_by_label)

    test_tds = TaskDataset(data=test_data, labels=test_labels, dname=tds.name, labels_meta=tds.labels_meta)
    train_tds = TaskDataset(data=train_data, labels=train_labels, dname=tds.name, labels_meta=tds.labels_meta)
    val_tds = TaskDataset(data=val_data, labels=val_labels, dname=tds.name, labels_meta=tds.labels_meta)

    return train_tds, test_tds, val_tds


def _get_split_amount(count_dict:dict, strategy:int, ratio:float, tolerance:float) -> dict:
    """

    :param count_dict: {class: sample_count}
    :param strategy: 1 for uniform, 0 to follow existing distribution
    :return:
    """
    out = {}
    if strategy == 0:
        balanced_cls_counts = [None if is_imbalanced_cls(cls, count_dict, tolerance) else count_dict[cls] for cls in count_dict]
        balanced_cls_counts = list(filter(lambda a: a != None, balanced_cls_counts))
        bavg = sum(balanced_cls_counts)/len(balanced_cls_counts)
        out = {cls: floor(bavg*ratio) if is_imbalanced_cls(cls, count_dict, tolerance) and count_dict[cls] > bavg
                                         or not is_imbalanced_cls(cls, count_dict, tolerance) else floor(count_dict[cls]*ratio) for cls in count_dict}
    else:
        out = {cls: floor(count_dict[cls]*ratio) for cls in count_dict}
    return out


def _retrieve_sample(data_dict:dict, count_dict_by_cls:dict):
    data, labels = [], []
    for label in data_dict:
        for _ in range(count_dict_by_cls[label]):
            index_to_pop = random.randint(0, len(data_dict[label])-1)
            data_to_pop = data_dict[label].pop(index_to_pop)
            data_to_pop = data_to_pop
            data.append(data_to_pop)
            labels.append(label)
    return data_dict, data, labels


if __name__ == "__main__":
    tds = TaskDataset()
    tds.load("dataset/banking77.tds")
    no_by_cls = count_label(tds.labels)
    no_by_cls = set_imbalance_by_cls(no_by_cls, 0.3, 0.3, 0.6)
    imbfactor = get_imb_factor(no_by_cls, 0.3)
    print(imbfactor)
