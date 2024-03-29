import spacy
import vars

from dataset_utils import preprocess_texts
from nltk.tokenize import sent_tokenize, word_tokenize

from Config import ModelConfig
from vars import (transformer_names,)

from model_utils import merge_trans_sent


class Tokenizer:
    def __init__(self, name, pretrained_model_name="", vocab_file=None):
        assert not (pretrained_model_name and vocab_file), \
            "Either provide vocab file for non-pretrained tokenizer or provide " \
            "the pretrained tokenizer without vocab file."

        self.name = name.lower()
        self.core = None

        if name == "spacy" or name == "spacy-sent":
            self.core = spacy.load("en_core_web_sm")

        elif name == "nltk":
            self.core = word_tokenize

        elif "bert" in name:
            from transformers import BertTokenizer
            self.core = BertTokenizer.from_pretrained(pretrained_model_name) \
                if pretrained_model_name else BertTokenizer(vocab_file, do_lower_case=True)

        elif "gpt2" in name:
            from transformers import GPT2Tokenizer
            self.core = GPT2Tokenizer.from_pretrained(pretrained_model_name) \
                if pretrained_model_name else GPT2Tokenizer(vocab_file, do_lower_case=True)

        elif "xlnet" in name:
            from transformers import XLNetTokenizer
            self.core = XLNetTokenizer.from_pretrained(pretrained_model_name) \
                if pretrained_model_name else XLNetTokenizer(vocab_file, do_lower_case=True)

        elif "roberta" in name:
            from transformers import RobertaTokenizer
            self.core = RobertaTokenizer.from_pretrained(pretrained_model_name) \
                if pretrained_model_name else RobertaTokenizer(vocab_file, do_lower_case=True)

    def __call__(self, texts):
        texts = preprocess_texts(texts)
        docs = None
        names = self.name.split("-")

        if names[-1] == "sent":
            if names[0] in transformer_names:
                docs = [[self.core(sent) for sent in sent_tokenize(doc)] for doc in texts]
                docs = merge_trans_sent(docs)
            elif names[0] == "nltk":
                docs = [[self.core(sent) for sent in sent_tokenize(doc)] for doc in texts]
            elif names[0] == "spacy":
                docs = self.core.pipe(texts, n_process=4, disable=["tok2vec", "transformer"])
                docs = [[[tok.text for tok in sent] for sent in doc.sents] for doc in docs]
        if names[0] == "spacy":
            docs = self.core.pipe(texts, n_process=4, disable=["tok2vec", "transformer"])
            docs = [[tok.text for tok in doc] for doc in docs]
        elif self.name == "nltk":
            docs = [self.core(text) for text in texts]
        elif self.name in transformer_names:
            docs = self.core(texts)
        return docs

    def __len__(self):
        if self.name in transformer_names:
            return len(self.core)
        else:
            return vars.cutoff+2


def main(config: ModelConfig):
    Model = None
    config = config()

    if config.model_name in ['gpt2', 'bert', 'xlnet', 'roberta']:
        if config.model_name == "gpt2":
            from classifiers.GPT2 import Model
        elif config.model_name == "bert":
            from classifiers.BERT import Model
        elif config.model_name == "xlnet":
            from classifiers.XLNet import Model
        elif config.model_name == "roberta":
            from classifiers.Roberta import Model

        if config.pretrained_model_name:
            model = Model.from_pretrained(
                config.pretrained_model_name, num_labels=config.num_labels)
            model.config.device = config.device
            if config.model_name == "roberta":
                model.config.max_position_embeddings = 512

        # Unify tokenizer length and model vocab size
        else:
            tokenizer = Tokenizer(config.tokenizer_name,
                                  pretrained_model_name=config.pretrained_tokenizer_name)

            config.vocab_size = len(tokenizer)
            model = Model(config)

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

    model.tokenizer = Tokenizer(config.tokenizer_name,
                                pretrained_model_name=config.pretrained_tokenizer_name)

    return model
