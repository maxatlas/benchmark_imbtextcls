import torch
import copy
import torch.nn as nn
import dill
import math

from utils import get_max_lengths


class TaskModel(nn.Module):
    def __init__(self, config):
        super(TaskModel, self).__init__()
        self.config = config

        emb_weights = torch.load(config.emb_path)

        vocab_size, self.emb_d = emb_weights['weight'].shape
        self.emb = nn.Embedding(vocab_size, self.emb_d)
        self.emb.load_state_dict(emb_weights)

        self.cls = nn.Linear(config.cls_hidden_size, config.num_labels)

        self.word_index = dill.load(open(config.word_index_path, "rb"))

        self.tokenizer = None

    def freeze_emb(self):
        self.emb.weight.requires_grad = False

    def unfreeze_emb(self):
        self.emb.weight.requires_grad = True

    def batch_train(self, texts, labels, label_names, loss_func):
        tokens = self.tokenizer(texts)

        if self.tokenizer.name in ["spacy", "spacy-sent", "nltk", "nltk-sent"]:
            if "sent" in self.tokenizer.name:
                token_ids = [[[self.word_index.get(word, 0) for word in sent] for sent in doc] for doc in tokens]
            else:
                token_ids = [[self.word_index.get(word, 0) for word in doc] for doc in tokens]
        else:
            token_ids = tokens['input_ids']

        logits, _ = self.forward(token_ids)

        label_ids = get_label_ids(labels, label_names)
        loss = loss_func(logits, label_ids)

        return loss

    def batch_eval(self, texts, labels, label_names):
        tokens = self.tokenizer(texts)

        if "sent" in self.tokenizer.name:
            token_ids = [[[self.word_index.get(word, 0) for word in sent] for sent in doc] for doc in tokens]
        else:
            token_ids = [[self.word_index.get(word, 0) for word in doc] for doc in tokens]

        with torch.no_grad():
            _, preds = self.forward(token_ids)
            if type(labels[0]) is list:
                preds = get_label_ids(preds, label_names).long()
        return preds


def get_label_ids(labels, label_names):
    assert type(labels) is int or (type(labels) is list and type(labels[0]) is int), \
        "Format of the label should be either int or list(int). " \
        "Please transform labels at build_dataset stage."

    label_ids = [len(label_names) * [0]] * len(labels)
    label_list = copy.deepcopy(labels)
    for i, labels in enumerate(label_list):
        if type(labels) is int:
            labels = [labels]
        for label in labels:
            label_ids[i][label] = 1
    label_ids = torch.tensor(label_ids)
    return label_ids.float()


def batch_train(self, texts, labels, label_names, loss_func):
    label_ids = get_label_ids(labels, label_names)
    logits, _ = self.forward(texts)

    loss = loss_func(logits, label_ids)

    return loss


def batch_eval(self, texts, labels, label_names):
    with torch.no_grad():
        _, preds = self.forward(texts)
        if type(labels[0]) is list:
            preds = get_label_ids(preds, label_names).long()
    return preds


def pad_seq(text_ids, length_limit=None):
    length_limit = math.inf if not length_limit else length_limit

    _, word_max_length = get_max_lengths(text_ids)
    word_max_length = word_max_length if word_max_length < length_limit else length_limit

    text_ids = [(doc + [0] * (word_max_length - len(doc)))
                [:word_max_length] for doc in text_ids]  # pad words
    text_ids = torch.tensor(text_ids)

    return text_ids


def pad_seq_to_length(text_ids, word_max_length):
    text_ids = [(doc + [0] * (word_max_length - len(doc)))
                [:word_max_length] for doc in text_ids]  # pad words
    text_ids = torch.tensor(text_ids)

    return text_ids


def pad_seq_han(text_ids):
    word_max_length, sent_max_length = get_max_lengths(text_ids)

    text_ids = [[(sent + [0] * (word_max_length - len(sent)))
                 [:word_max_length] for sent in doc] for doc in text_ids]  # pad words
    text_ids = [(doc + [[0] * word_max_length] * (sent_max_length - len(doc)))
                [:sent_max_length] for doc in text_ids]  # pad sentences
    text_ids = torch.tensor(text_ids)

    return text_ids
