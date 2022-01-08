import spacy
import torch
import torch.nn as nn

from utils import preprocess_texts
from nltk.tokenize import sent_tokenize, word_tokenize


class Tokenizer:
    def __init__(self, name, pretrained_model_name="", vocab_file=None):
        assert not (pretrained_model_name and vocab_file), \
            "Either provide vocab file for non-pretrained tokenizer or provide " \
            "the pretrained tokenizer without vocab file."
        assert name in ["spacy", "nltk", "nltk_sent", "bert", "gpt2", "xlnet", "roberta"], \
            "Tokenizer name needs to be one of the following:" \
            "\n\t\tspacy\n\t\tnltk\n\t\tnltk_sent\n\t\tbert\n\t\tgpt2\n\t\txlnet\n\t\troberta"

        self.name = name.lower()
        self.core = None

        if name == "spacy":
            self.core = spacy.load("en_core_web_sm")
        elif name == "nltk":
            self.core = word_tokenize

        elif name == "bert":
            from transformers import BertTokenizer
            self.core = BertTokenizer.from_pretrained(pretrained_model_name) \
                if pretrained_model_name else BertTokenizer(vocab_file, do_lower_case=True)

        elif name == "gpt2":
            from transformers import GPT2Tokenizer
            self.core = GPT2Tokenizer.from_pretrained(pretrained_model_name) \
                if pretrained_model_name else GPT2Tokenizer(vocab_file, do_lower_case=True)

        elif name == "xlnet":
            from transformers import XLNetTokenizer
            self.core = XLNetTokenizer.from_pretrained(pretrained_model_name) \
                if pretrained_model_name else XLNetTokenizer(vocab_file, do_lower_case=True)

        elif name == "roberta":
            from transformers import RobertaTokenizer
            self.core = RobertaTokenizer.from_pretrained(pretrained_model_name) \
                if pretrained_model_name else RobertaTokenizer(vocab_file, do_lower_case=True)

    def __call__(self, texts):
        texts = preprocess_texts(texts)
        if self.name is "spacy":
            return self.core.pipe(texts, n_process=2)
        elif self.name is "nltk_sent":
            return [[word_tokenize(sent) for sent in sent_tokenize(doc)] for doc in texts]
        elif self.name is "nltk":
            return [self.core(text) for text in texts]
        elif self.name in ["bert", "gpt2", "xlnet", "roberta"]:
            return self.core(texts)


def main(config):
    config = config()

    if config.model_name in ['gpt2', 'bert', 'xlnet', 'roberta']:
        pretrained_model_name = ""
        if config.model_name == "gpt2":
            from classifiers.GPT2 import Model
            pretrained_model_name = "gpt2"
        elif config.model_name == "bert":
            pretrained_model_name = "bert-base-uncased"
            from classifiers.BERT import Model
        elif config.model_name == "xlnet":
            pretrained_model_name = "xlnet-large-cased"
            from classifiers.XLNet import Model
        elif config.model_name == "roberta":
            pretrained_model_name = "roberta-base"
            from classifiers.Roberta import Model

        model = Model(config).from_pretrained(
            pretrained_model_name, num_labels=config.num_labels)

    else:
        if config.model_name == "cnn":
            from classifiers.CNN import Model
        elif config.model_name == "rcnn":
            from classifiers.RCNN import Model
        elif config.model_name == "lstm":
            from classifiers.LSTM import Model
        elif config.model_name == "lstmattn":
            from classifiers.LSTMattn import Model
        elif config.model_name == "mlp":
            from classifiers.MLP import Model
        elif config.model_name == "han":
            from classifiers.HAN import Model

        model = Model(config)

    if config.emb_layer_path:
        weight = torch.load(config.emb_layer_path)
        emb = nn.Embedding(*weight['weight'].shape)
        emb.load_state_dict(weight)

        model.emb = emb

    model.tokenizer = Tokenizer(config.tokenizer_name,
                                pretrained_model_name=config.pretrained_model_name)

    return model

