from transformers import (
    GPT2Tokenizer, XLNetTokenizer, BertTokenizer, RobertaTokenizer,
    GPT2Config, BertConfig, XLNetConfig, RobertaConfig)
from Classifier import (
    GPT2, BERT, XLNet, Roberta,
    LSTMClassifier
)


class ModelConfig:
    def __init__(self):
        self.n_positions = None


class faketkn:
    def __init__(self):
        self.model_max_length = None

    def __call__(self, data):
        return {'input_ids': data}


models = {
    "gpt": {
        "tokenizer": GPT2Tokenizer.from_pretrained("gpt2"),
        "model": GPT2,
        "config": GPT2Config
    },
    "xlnet":{
        "tokenizer": XLNetTokenizer.from_pretrained("xlnet-large-cased", do_lower_case=True),
        "config": XLNetConfig,
        "model": XLNet,
    },
    "bert":{
        "tokenizer": BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True),
        "config": BertConfig,
        "model": BERT,
    },

    "roberta":{
        "tokenizer": RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True),
        "config": RobertaConfig,
        "model": Roberta,
    },

    "lstm":{
        "tokenizer": faketkn(),
        "model": LSTMClassifier,
        "config": ModelConfig
    }
}