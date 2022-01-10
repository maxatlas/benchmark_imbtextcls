import copy
import math


def preprocess_text(text: str):
    text = text.lower()
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")
    text = text.replace("//", " ")
    text = text.replace("\\", " ")
    return text


def preprocess_texts(texts:list):
    return [preprocess_text(text) for text in texts]


def set_imb_count_dict(count_dict: dict, tolerance: float, threshold: float,
                       cls_ratio_to_imb, sample_ratio_to_imb, balance_strategy: str = None,
                       split_ratio: str = "0.75/0.20/0.05"):
    split_ratio = [float(i) for i in split_ratio.split("/")]

    if not balance_strategy:
        train_no_by_label = {lb: math.floor(count_dict[lb] * split_ratio[0])
                             for lb in count_dict}
    else:
        train_no_by_label = _get_split_amount_by_strategy(count_dict, balance_strategy, split_ratio[0],
                                                          tolerance, threshold)

    test_no_by_label = {lb: math.floor((count_dict[lb] - train_no_by_label[lb]) * split_ratio[1])
                        for lb in count_dict}
    val_no_by_label = {lb: count_dict[lb] - test_no_by_label[lb] - train_no_by_label[lb]
                       for lb in count_dict}
    if not balance_strategy and not is_imbalanced_ds(count_dict):
        train_no_by_label = set_imbalance_by_cls(train_no_by_label, tolerance, threshold,
                                                 cls_ratio_to_imb, sample_ratio_to_imb)

    print("Train/Test/Val split:")
    print(train_no_by_label)
    print(test_no_by_label)
    print(val_no_by_label)

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


def _get_split_amount_by_strategy(count_dict: dict, strategy: str, ratio: float, tolerance: float,
                                  threshold: float) -> dict:
    """

    :param count_dict: {class: sample_count}
    :param strategy: 1 for uniform, 0 to follow existing distribution
    :return:
    """
    out = {}
    if strategy == "undersample":
        print(count_dict)
        balanced_cls_counts = [0 if is_imbalanced_cls(cls, count_dict, tolerance) else
                               count_dict[cls] for cls in count_dict]
        balanced_cls_counts = list(filter(lambda a: a != 0, balanced_cls_counts))
        bavg = sum(balanced_cls_counts) / len(balanced_cls_counts)
        imb_cls_counts = [count_dict[cls] if is_imbalanced_cls(cls, count_dict, tolerance)
                                else 0 for cls in count_dict]
        imb_small_cls_counts = [0 if count > bavg else count for count in imb_cls_counts]
        imb_small_cls_counts = list(filter(lambda a: a != 0, imb_small_cls_counts))
        isavg = sum(imb_small_cls_counts) / len(imb_small_cls_counts)
        out = {cls: math.floor(isavg * ratio) if is_imbalanced_cls(cls, count_dict, tolerance)
                                                 and count_dict[cls] > isavg
                                                 or not is_imbalanced_cls(cls, count_dict, tolerance)
                                              else math.floor(count_dict[cls] * ratio) for cls in count_dict}
        # if cls is balanced or too big, take balanced avg * test_split_ratio
        # if cls is imbalanced and too small, take its size * test_split_ratio.

        assert not is_imbalanced_ds(out, tolerance, threshold) or \
               not is_imbalanced_ds(out, tolerance - 0.1, threshold - 0.1)
    elif strategy == "uniform":
        balanced_cls_counts = [None if is_imbalanced_cls(cls, count_dict, tolerance) else count_dict[cls] for cls in
                               count_dict]
        balanced_cls_counts = list(filter(lambda a: a is not None, balanced_cls_counts))
        bavg = sum(balanced_cls_counts) / len(balanced_cls_counts)
        out = {cls: math.floor(bavg * ratio) if is_imbalanced_cls(cls, count_dict, tolerance) and count_dict[cls] > bavg
                                           or not is_imbalanced_cls(cls, count_dict, tolerance) else math.floor(
            count_dict[cls] * ratio) for cls in count_dict}

        assert not is_imbalanced_ds(out, tolerance, threshold) or \
               not is_imbalanced_ds(out, tolerance - 0.1, threshold - 0.1)
    elif strategy == "oversample":
        pass
    return out


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


def get_max_lengths(input_ids):
    word_length_list = []
    sent_length_list = []

    for sents in input_ids:
        if type(sents[0]) is list:
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
