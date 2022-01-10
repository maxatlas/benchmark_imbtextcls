import spacy

from utils import preprocess_texts
from nltk.tokenize import sent_tokenize, word_tokenize


def save_all_transformer_emb_layer():
    from Config import ModelConfig
    from model_utils import save_transformer_emb
    names = [("bert", "bert-base-uncased"),
           ("xlnet", "xlnet-base-cased"),
           ("gpt2", "gpt2"),
           ("roberta", "roberta-base")]
    for model_name, pretrained_name in names[-1:]:
        print("\n"+model_name)
        mc = ModelConfig(model_name, 2, pretrained_model_name=pretrained_name)
        model = main(mc)
        save_transformer_emb(model, model_name)
        print("Done.")


class Tokenizer:
    def __init__(self, name, pretrained_model_name="", vocab_file=None):
        assert not (pretrained_model_name and vocab_file), \
            "Either provide vocab file for non-pretrained tokenizer or provide " \
            "the pretrained tokenizer without vocab file."
        assert name in ["spacy", "spacy-sent", "nltk", "nltk-sent", "bert", "gpt2", "xlnet", "roberta"], \
            "Tokenizer name needs to be one of the following:" \
            "\n\t\tspacy\n\t\tspacy_sent\n\t\tnltk\n\t\tnltk_sent\n\t\tbert\n\t\tgpt2\n\t\txlnet\n\t\troberta"

        self.name = name.lower()
        self.core = None

        if name == "spacy" or name == "spacy-sent":
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
        docs = None
        if self.name == "spacy":
            docs = self.core.pipe(texts, n_process=2)
            docs = [[tok.text for tok in doc] for doc in docs]
        elif self.name == "spacy-sent":
            docs = self.core.pipe(texts, n_process=2)
            docs = [[[tok.text for tok in sent] for sent in doc.sents] for doc in docs]
        elif self.name == "nltk-sent":
            docs = [[word_tokenize(sent) for sent in sent_tokenize(doc)] for doc in texts]
        elif self.name == "nltk":
            docs = [self.core(text) for text in texts]
        elif self.name in ["bert", "gpt2", "xlnet", "roberta"]:
            docs = self.core(texts)
        return docs


def main(config):
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
            model = Model(config).from_pretrained(
                config.pretrained_model_name, num_labels=config.num_labels)
            if config.model_name == "roberta":
                model.config.max_position_embeddings = 512
        else:
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
