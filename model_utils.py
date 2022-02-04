import torch
import copy
import torch.nn as nn
import dill
import math

from dataset_utils import get_max_lengths


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.layer = nn.Linear(config.hidden_size, config.num_labels).to(config.device)
        # can add layernorm, dropout etc. so that it's cleaner on the final model's side.

    def forward(self, x):
        out = self.layer(x)
        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    @staticmethod
    def forward(x):
        return x


class Identity2(nn.Module):
    def __init__(self):
        super(Identity2, self).__init__()

    @staticmethod
    def forward(x, y):
        out = x + y
        return out


class TaskModel(nn.Module):
    def __init__(self, config):
        super(TaskModel, self).__init__()
        self.config = config

        emb_obj = dill.load(open(config.emb_path, "rb"))
        self.emb = emb_obj['word'].to(config.device)

        self.pos_emb = emb_obj.get('pos')

        if self.pos_emb:
            self.pos_emb.to(config.device)

        vocab_size, self.emb_d = self.emb.weight.shape

        self.cls = nn.Linear(config.cls_hidden_size, config.num_labels).to(config.device)

        if "word_index_path" in config.to_dict():
            self.word_index = dill.load(open(config.word_index_path, "rb"))

        self.tokenizer = None

    def freeze_emb(self):
        self.emb.weight.requires_grad = False

    def unfreeze_emb(self):
        self.emb.weight.requires_grad = True

    def _get_token_ids(self, texts):
        # For debugging
        tokens = self.tokenizer(texts)

        if self.tokenizer.name in ["spacy", "spacy-sent", "nltk", "nltk-sent"]:
            if "sent" in self.tokenizer.name:
                token_ids = [[[self.word_index.get(word, 0) for word in sent] for sent in doc] for doc in tokens]
            else:
                token_ids = [[self.word_index.get(word, 0) for word in doc] for doc in tokens]
        else:
            token_ids = tokens['input_ids']
        return token_ids

    def batch_train(self, texts, label_ids, label_names, loss_func, multi_label):
        tokens = self.tokenizer(texts)

        if self.tokenizer.name in ["spacy", "spacy-sent", "nltk", "nltk-sent"]:
            if "sent" in self.tokenizer.name:
                token_ids = [[[self.word_index.get(word, 0) for word in sent] for sent in doc] for doc in tokens]
            else:
                token_ids = [[self.word_index.get(word, 0) for word in doc] for doc in tokens]
            logits, _ = self.forward(token_ids)
        else:
            token_ids = tokens['input_ids']
            logits, _ = self.forward(token_ids)

        label_ids = label_ids.float()

        if multi_label:
            loss = loss_func(logits.view(-1, len(label_names)),
                             label_ids.view(-1, len(label_names)).
                             type_as(logits.view(-1, len(label_names)))
                             )
        else:
            loss = loss_func(logits, label_ids.to(self.config.device))

        return loss

    def batch_eval(self, texts, labels, label_names, multi_label=False):
        tokens = self.tokenizer(texts)

        if self.tokenizer.name in ["spacy", "spacy-sent", "nltk", "nltk-sent"]:
            if "sent" in self.tokenizer.name:
                token_ids = [[[self.word_index.get(word, 0) for word in sent] for sent in doc] for doc in tokens]
            else:
                token_ids = [[self.word_index.get(word, 0) for word in doc] for doc in tokens]
        else:
            token_ids = tokens['input_ids']

        with torch.no_grad():
            logits, preds = self.forward(token_ids)
            probs = torch.sigmoid(logits)
            if multi_label:
                preds = (probs > 0.5).long()
            else:
                preds = get_label_ids(preds.tolist(), label_names).long()
            labels = torch.tensor(labels).long()
        return probs, preds, labels


def get_label_ids(labels, label_names):
    assert type(labels) is int or (type(labels) is list and type(labels[0]) is int or list), \
        "Format of the label should be either int or list(int). " \
        "Please transform labels at build_dataset stage."

    label_ids = torch.zeros(len(labels), len(label_names))
    label_list = copy.deepcopy(labels)
    for i, labels in enumerate(label_list):
        if type(labels) is int:
            labels = [labels]
        for label in labels:
            label_ids[i][label] = 1
    return label_ids


def batch_train(self, texts, label_ids, label_names, loss_func, multi_label):
    label_ids = label_ids.float()
    label_ids = label_ids.to(self.config.device)
    logits, _ = self.forward(texts)

    if multi_label:
        loss = loss_func(logits.view(-1, len(label_names)),
                         label_ids.view(-1, len(label_names)).
                         type_as(logits.view(-1, len(label_names)))
                         )
    else:
        loss = loss_func(logits, label_ids)
    return loss


def batch_eval(self, texts, labels, label_names, multi_label):
    with torch.no_grad():
        logits, preds = self.forward(texts)
        probs = torch.sigmoid(logits)
        if multi_label:
            preds = (probs > 0.5).long()
        else:
            preds = get_label_ids(preds.tolist(), label_names)
        labels = torch.tensor(labels).long()
    return probs, preds, labels


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


def load_transformer_emb(model, i=0):
    weight = list(model.parameters())[i]
    emb = nn.Embedding(*weight.shape)
    emb.load_state_dict({"weight": weight})

    return emb.state_dict()


def save_transformer_emb(model, model_name):
    i = 1 if model_name == "xlnet" else 0
    emb = load_transformer_emb(model, i)
    torch.save(emb, "params/emb_layer_%s" % model_name)


def merge_trans_sent(res):
    d = {}
    keys = res[0][0].keys()
    for key in keys:
        d[key] = [[sent[key] for sent in doc] for doc in res]
    return d
