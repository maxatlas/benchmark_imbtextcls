from transformers import (
    GPT2Tokenizer, XLNetTokenizer, BertTokenizer, RobertaTokenizer,
    GPT2Config, BertConfig, XLNetConfig, RobertaConfig)
from Classifier import (
    GPT2, BERT, XLNet, Roberta,
    LSTM, CNN, RCNN, HAN
)


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
    def __init__(self, tokenizer=None):
        self.model_max_length = None
        self.tokenizer = tokenizer

    def __len__(self):
        return self.model_max_length

    def __call__(self, data):
        return {'input_ids': data}


models = {
    "gpt": {
        "tokenizer": GPT2Tokenizer.from_pretrained("gpt2"),
        "model": GPT2,
        "config": GPT2Config()
    },
    "xlnet":{
        "tokenizer": XLNetTokenizer.from_pretrained("xlnet-large-cased", do_lower_case=True),
        "config": XLNetConfig(),
        "model": XLNet,
    },
    "bert":{
        "tokenizer": BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True),
        "config": BertConfig(),
        "model": BERT,
    },

    "roberta":{
        "tokenizer": RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True),
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
        "tokenizer": faketkn(),
        "model": HAN,
        "config": ModelConfig()
    }
}