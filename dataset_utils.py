import copy
import math
import numpy as np
import pandas as pd
from collections import defaultdict


def get_head_class(count_dict):
    """
    Return list of class names and counts that are the head classes.
    :param count_dict:
    :return:
    """
    out = []
    l = sorted(list(count_dict.items()), key=lambda x:x[1], reverse=True)
    for item in l[1:]:
        if item[1] > l[0][1] * 0.9:
            out.append(item)
        else:
            break
    return out


def get_tail_class(count_dict):
    """
    Return list of class names and counts that are the tail class.
    :param labels:
    :return:
    """
    out = []
    l = sorted(list(count_dict.items()), key=lambda x: x[1])
    for item in l[1:]:
        if item[1] < l[0][1] * 1.1:
            out.append(item)
        else:
            break
    return out


def get_count_dict(labels):
    out = defaultdict(int)
    for label in labels:
        if type(label) == int or type(label) == str:
            out[label] += 1
        else:
            for l in label:
                out[l] += 1
    return out


def oversample(df, target_count_dict: dict, label_field: str):
    """
    logic: how much more samples each class needs. have a dictionary of it, and add samples accordingly
    select a row - replicate
    :param label_field:
    :param target_count_dict:
    :param df:
    :return:
    """
    for key, value in target_count_dict.items():
        if value > 0:
            ids = []
            # Sample n rows by class and add it to ids
            cls_data, to_add = df.loc[df[label_field] == key], None
            if value < len(cls_data):
                ids.extend(cls_data.sample(value).index)
                to_add = df.loc[ids]
            # when sample amount > total data length of the class.
            else:
                to_add = pd.concat([cls_data] * (value // len(cls_data)) +
                                   [cls_data[: value % len(cls_data)]],
                                   ignore_index=True)

            if to_add is not None:
                df = pd.concat([df, to_add], ignore_index=True)

    return df


def undersample(df, target_count_dict: dict, label_field: str):
    """
    logic: how much more samples each class needs. have a dictionary of it, and add samples accordingly

    select a row - delete
    :param label_field:
    :param target_count_dict:
    :param df:
    :return:
    """
    ids = []
    for key, value in target_count_dict.items():
        # Sample n rows by class and add it to ids
        if value < 0:
            ids.extend(df.loc[df[label_field] == key].sample(abs(value)).index)
    df = df.drop(ids)
    return df


def preprocess_text(text: str):
    text = text.lower()
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")
    text = text.replace("//", " ")
    text = text.replace("\\", " ")
    return text


def preprocess_texts(texts:list):
    return [preprocess_text(text) for text in texts]


def get_label_ids(labels, label_names):
    label_ids = np.array(len(label_names) * [0])
    if type(labels) is int:
        labels = [labels]
    for i, label in enumerate(labels):
        label_ids[label] = 1
    return label_ids


def set_imb_count_dict(count_dict: dict, tolerance: float, threshold: float,
                       cls_ratio_to_imb, sample_ratio_to_imb,
                       split_ratio: str = "0.75/0.20/0.05",
                       make_it_imbalanced: bool = True):
    split_ratio = [float(i) for i in split_ratio.split("/")]

    train_no_by_label = {lb: math.floor(count_dict[lb] * split_ratio[0]) for lb in count_dict}

    test_no_by_label = {lb: math.floor((count_dict[lb] - train_no_by_label[lb]) * split_ratio[0])
                        for lb in count_dict}
    val_no_by_label = {lb: count_dict[lb] - train_no_by_label[lb] - test_no_by_label[lb]
                       for lb in count_dict}
    if make_it_imbalanced and not is_imbalanced_ds(count_dict):
        train_no_by_label = set_imbalance_by_cls(train_no_by_label, tolerance, threshold,
                                                 cls_ratio_to_imb, sample_ratio_to_imb)

    return train_no_by_label, test_no_by_label, val_no_by_label


def is_imbalanced_ds(no_by_cls: dict, tolerance=0.2, threshold=0.5):
    """

    Imbalanced classes > threshold will be deemed as imbalanced.
    :param no_by_cls:
    :param tolerance:
    :param threshold:
    :return:
    """
    imb_factor = get_imb_factor(no_by_cls, tolerance=tolerance)
    if imb_factor > threshold:
        return True
    return False


def is_imbalanced_cls(cls: str, count_dict: dict, tolerance=0.2):
    avg = sum(count_dict.values()) / len(count_dict)
    return count_dict[cls] / avg > (1 + tolerance) or count_dict[cls] / avg < (1 - tolerance)


def resample(df, label_field, balance_strategy: str):
    """

    :param labels:
    :param balance_strategy:
    :return:
    """
    if balance_strategy:
        labels = df[label_field]
        count_dict, target_class = get_count_dict(labels), None

        if "oversampl" in balance_strategy:
            target_class = get_head_class(count_dict)  # head

        elif "undersampl" in balance_strategy:
            target_class = get_tail_class(count_dict)  # tail
        avg = sum([item[1] for item in target_class]) / len(target_class)
        avg = int(avg)
        new_values = list(avg - np.array(list(count_dict.values())))

        keys = list(count_dict.keys())
        resample_dict = dict(zip(keys, new_values))

        print("Resampling dict:")
        print(resample_dict)

        out = df
        if "oversampl" in balance_strategy:
             out = oversample(df, resample_dict, label_field)

        elif "undersampl" in balance_strategy:
            out = undersample(df, resample_dict, label_field)

        return out
    else:
        return df


def get_imb_factor(no_by_cls: dict, tolerance=0.2):
    imb_classes = 0
    counts = sorted(no_by_cls.values())
    avg = sum(counts) / len(no_by_cls)
    for count in counts:
        if count > avg * (1 + tolerance) or count < avg * (1 - tolerance):
            imb_classes += 1
    return imb_classes / len(no_by_cls)


def set_imbalance_by_cls(no_by_cls: dict, tolerance, threshold, cls_ratio_to_imb=0.75, sample_ratio_to_imb=0.5):
    # choose a ratio of balanced classes, reduce amount. assert is_imbalanced_ds
    # recursively reduce until imbalanced
    out = copy.deepcopy(no_by_cls)
    imb_no = math.floor(len(no_by_cls) * cls_ratio_to_imb)
    # choose this many classes to transform from balance to imbalance
    imb_no = 1 if not imb_no else imb_no
    # if there are 2 classes in total, the last var could be 0, so manually set to 1

    cls_count = sorted([(cls, no_by_cls[cls]) for cls in no_by_cls], key=lambda x: x[1], reverse=True)
    # cls, and its sample size
    bcls_count = [None if is_imbalanced_cls(cls, no_by_cls, tolerance) else (cls, count) for cls, count in cls_count]
    bcls_count = list(filter(lambda x: x is not None, bcls_count))[-imb_no:]
    # get the cls name and its sample size to transform from balanced -> imbalanced
    bcls_count = [(cls, math.floor(count * sample_ratio_to_imb)) for cls, count in bcls_count]
    # reduce each class's sample size.

    for cls, count in bcls_count:
        out[cls] = count
        # store in output dictionary

    print("Split result for training set:")
    print(out)

    print("Checking if the new split is imbalanced ...")
    assert is_imbalanced_ds(out, tolerance, threshold)
    print("Checked.")

    return out


def get_max_lengths(input_ids: list):
    word_length_list = []
    sent_length_list = []

    for sents in input_ids:
        if sents and type(sents[0]) is list:
            for words in sents:
                word_length_list.append(len(words))
        sent_length_list.append(len(sents))

    sorted_word_length = sorted(word_length_list)
    sorted_sent_length = sorted(sent_length_list)

    sent_max_length = sorted_sent_length[int(0.9*len(sorted_sent_length))]

    word_max_length = None
    if sorted_word_length:
        word_max_length = sorted_word_length[int(0.9*len(sorted_word_length))]

    return word_max_length, sent_max_length
