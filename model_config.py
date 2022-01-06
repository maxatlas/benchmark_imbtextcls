from transformers import (
    GPT2Tokenizer, XLNetTokenizer, BertTokenizer, RobertaTokenizer,
    GPT2Config, BertConfig, XLNetConfig, RobertaConfig)
from Classifier import (
    GPT2, BERT, XLNet, Roberta,
    LSTM, CNN, RCNN, HAN,
    LSTMattn, MLP
)
import itertools


class ModelConfig():
    def __init__(self, **kwargs):
        self.hidden_size = 10
        self.hidden_dropout_prob = 0.1
        for key, value in kwargs:
            self.__setattr__(key, value)

    def from_dict(self, d):
        for key, value in d.items():
            self.__setattr__(key, value)
        return self

    def to_dict(self):
        return self.__dict__


class faketkn:
    def __init__(self, tokenizer=[]):
        self.model_max_length = None
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tokenizer)

    def __call__(self, data):
        # merge sentences for each document.
        if self.tokenizer: return self.tokenizer(data)

        data = [list(itertools.chain(*doc)) for doc in data]
        return {'input_ids': data}


class hantkn(faketkn):
    def __call__(self, data):
        return {'input_ids': data}


models = {
    "gpt": {
        "tokenizer": faketkn(GPT2Tokenizer.from_pretrained("gpt2")),
        "model": GPT2,
        "config": GPT2Config()
    },
    "xlnet":{
        "tokenizer": faketkn(XLNetTokenizer.from_pretrained("xlnet-large-cased", do_lower_case=True)),
        "config": XLNetConfig(),
        "model": XLNet,
    },
    "bert":{
        "tokenizer": faketkn(BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)),
        "config": BertConfig(),
        "model": BERT,
    },

    "roberta":{
        "tokenizer": faketkn(RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)),
        "config": RobertaConfig(),
        "model": Roberta,
    },

    "lstm":{
        "tokenizer": faketkn(),
        "model": LSTM,
        "config": ModelConfig()
    },

    "cnn":{
        "tokenizer": faketkn(),
        "model": CNN,
        "config": ModelConfig()
    },

    "rcnn":{
        "tokenizer": faketkn(),
        "model": RCNN,
        "config": ModelConfig()
    },

    "han":{
        "tokenizer": hantkn(),
        "model": HAN,
        "config": ModelConfig()
    },

    "lstmattn":{
        "tokenizer": faketkn(),
        "model": LSTMattn,
        "config": ModelConfig()
    },

    "mlp": {
        "tokenizer": faketkn(),
        "model": MLP,
        "config": ModelConfig()
    },
}