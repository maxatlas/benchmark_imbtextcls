import copy
from hashlib import sha256

import vars
from vars import (model_names,
                  transformer_names,
                  parameter_folder)


def get_tokenizer_name(pretrained_tokenizer_name):
    return pretrained_tokenizer_name.split("-")[0]


def transformer_config(self, word_max_length, dropout, activation_function):
    if self.model_name == "bert" or self.model_name == "roberta":
        # only record max length when tokenizer isn't pretrained.
        if not self.pretrained_tokenizer_name:
            self.max_position_embeddings = word_max_length
        self.hidden_dropout_prob = dropout
        self.classifier_dropout = dropout
        self.attention_probs_dropout_prob = dropout

    elif self.model_name == "gpt2":
        # only record max length when tokenizer isn't pretrained.
        if not self.pretrained_tokenizer_name:
            self.n_positions = word_max_length

        self.resid_pdrop = dropout
        self.embd_pdrop = dropout
        self.attn_pdrop = dropout

    elif self.model_name == "xlnet":
        self.ff_activation = activation_function


def tokenizer_config(self):
    return self


def model_config(self, padding, dilation, stride, filters, hidden_size):
    if self.model_name == "cnn":
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.filters = filters

    elif self.model_name == "han":
        self.word_hidden_size = hidden_size
        self.sent_hidden_size = hidden_size


class DataConfig:
    def __init__(self,
                 huggingface_dataset_name,
                 label_field: str,
                 text_fields: list,
                 cls_ratio_to_imb: float,
                 sample_ratio_to_imb: float,
                 split_ratio="0.75/0.20/0.05",
                 balance_strategy=None,
                 threshold=0.6, tolerance=0.3,
                 test=None,
                 limit=200_000,
                 multi_label=False,
                 make_it_imbalanced: bool = True):
        assert not huggingface_dataset_name or type(huggingface_dataset_name) is list or tuple, \
            "huggingface_dataset_name wrongly formatted. A valid example: (glue, sst) or [glue, sst]"
        assert (type(split_ratio) is str and all([float(e) for e in split_ratio.split("/")])
                and len(split_ratio.split("/")) > 1), \
            "Wrong format for split_ratio. Should be 'train/test/val' or 'train/test'. " \
            "A valid example: '0.75/0.2/0.05'"
        assert not balance_strategy or balance_strategy in ["undersample", "oversample", "", None], \
            "Train set split strategy could be None, 'undersample', 'oversample'. Implement if need more."

        self.huggingface_dataset_name = huggingface_dataset_name
        # Path to glove/word2vec/fasttext. None if won't transform from token to token index.
        self.label_field = label_field
        self.text_fields = text_fields

        self.split_ratio = split_ratio
        # train/test/val split
        self.imb_tolerance = tolerance
        # class of sample size more or less than :tolerance: ratio from avg is deemed as imbalanced.
        self.imb_threshold = threshold
        # dataset of more than :threshold: ratio of imbalanced classes is deemed as imbalanced.

        self.cls_ratio_to_imb = cls_ratio_to_imb
        self.sample_ratio_to_imb = sample_ratio_to_imb
        # Both none if dataset already imbalanced.

        self.balance_strategy = balance_strategy

        self.test = test
        self.limit = limit
        self.multi_label = multi_label

        # Turn a balanced dataset imbalanced
        self.make_it_imbalanced = make_it_imbalanced

    def to_dict(self):
        return self.__dict__

    def idx(self):
        return sha256(str(self.to_dict()).encode('utf-8')).hexdigest()


class ModelConfig:
    def __init__(self, model_name,
                 num_labels,
                 tokenizer_name=None,
                 device=None,
                 word_max_length=None,
                 hidden_size=100,
                 word_index_path=None,
                 emb_path=None,
                 n_layers=1,
                 activation_function="gelu",
                 dropout=0.1,
                 padding=0,
                 dilation=1,
                 stride=1,
                 filters=(2, 3, 4, 5),
                 pretrained_model_name="",
                 pretrained_tokenizer_name="",
                 lr=0.001,
                 disable_output=True,
                 disable_selfoutput=True,
                 disable_intermediate=True,
                 add_pooling_layer=False,
                 n_heads=1,
                 qkv_size=None,
                 ):

        self.model_name = model_name.lower()
        self.pretrained_model_name = pretrained_model_name

        self.tokenizer_name = tokenizer_name
        self.pretrained_tokenizer_name = pretrained_tokenizer_name

        self.lr = lr
        self.num_layers = n_layers

        assert num_labels, "Must specify number of labels (num_labels)."
        assert self.model_name in model_names, \
            "Model name should be any of the following:" \
            "\n\t\tBert\n\t\tXLNet" \
            "\n\t\tRoBERTa\n\t\tGPT2\n\t\tLSTM" \
            "\n\t\tlstmattn (LSTM with attention) " \
            "\n\t\tCNN\n\t\tRCNN\n\t\tHAN\n\t\tMLP"

        self.num_labels = num_labels
        self.device = device


        if pretrained_model_name:
            assert any([transformer_name in pretrained_model_name
                        for transformer_name in transformer_names]), "Pretrained model name is not valid."

            self.tokenizer_name = self.model_name
            self.pretrained_model_name = pretrained_model_name
            self.pretrained_tokenizer_name = pretrained_model_name

            # Fix a Roberta pretrained bug.
            if self.model_name == "roberta":
                self.max_position_embeddings = 2 if not word_max_length else word_max_length
                self.vocab_size = 2 if not word_max_length else word_max_length

        else:  # Customized models or transformers
            if model_name in transformer_names:  # Customized transformers
                self.disable_intermediate = disable_intermediate
                self.disable_output = disable_output
                self.disable_selfoutput = disable_selfoutput
                self.cls_hidden_size = hidden_size

                qkv_size = 768 if not qkv_size else qkv_size
                if qkv_size != 768:
                    self.qkv_size = qkv_size

                transformer_config(self, word_max_length, dropout, activation_function)

                if model_name in ["bert", "roberta"]:
                    self.add_pooling_layer = add_pooling_layer
                    self.num_hidden_layers = n_layers
                    self.num_attention_heads = n_heads
                elif model_name in ["gpt2", "xlnet"]:
                    self.n_head = n_heads
                    self.n_layer = n_layers

                if pretrained_tokenizer_name:  # Customized transformers with pretrained transformer tokenizers.
                    assert any([transformer_name in pretrained_tokenizer_name
                                for transformer_name in transformer_names]), "Pretrained tokenizer name is not valid."

                    self.tokenizer_name = get_tokenizer_name(pretrained_tokenizer_name)

                    word_max_length = 512

                    if self.tokenizer_name == "bert" or self.tokenizer_name == "roberta":
                        emb_d = 768
                    elif self.tokenizer_name == "gpt2" or self.tokenizer_name == "xlnet":
                        emb_d = 768

                    if model_name == "bert" or model_name == "roberta":
                        self.hidden_size = emb_d
                        self.max_position_embeddings = word_max_length
                    elif model_name == "gpt2":
                        self.n_embd = emb_d
                        self.n_positions = word_max_length
                    elif model_name == "xlnet":
                        self.d_model = emb_d

                    if not emb_path:
                        self.emb_path = vars.parameter_folder + "/emb_layer_%s" % \
                                        (get_tokenizer_name(pretrained_tokenizer_name))

                else:
                    raise NotImplementedError("Customized transformer tokenizers not implemented.")
                    # tokenizer_config(self)
            else:  # Customized non transformer models
                model_config(self, padding, dilation, stride, filters, hidden_size)

                self.word_max_length = word_max_length
                self.emb_path = emb_path
                self.cls_hidden_size = hidden_size

                self.dropout = dropout
                self.activation = activation_function
                self.tokenizer = None

                if pretrained_tokenizer_name:  # with pretrained tokenizer.
                    assert any([transformer_name in pretrained_tokenizer_name
                                for transformer_name in transformer_names]), "Pretrained tokenizer name is not valid."
                    self.tokenizer_name = get_tokenizer_name(pretrained_tokenizer_name)
                    if model_name == "han":
                        self.tokenizer_name = self.tokenizer_name + "-sent"
                    self.emb_path = "%s/emb_layer_%s" % (parameter_folder,
                                                         get_tokenizer_name(pretrained_tokenizer_name))

                else:
                    if not word_index_path:
                        word_index_path = "% s / word_index" % parameter_folder
                    self.word_index_path = word_index_path
                    if tokenizer_name in transformer_names:
                        raise NotImplementedError("Customized transformer tokenizers not implemented.")
                        # tokenizer_config(self)
                    else:  # Customized models with customized tokenizer.
                        assert emb_path and tokenizer_name and word_index_path, \
                            "Customized model requires word_max_length," \
                            "\n\t\temb_path,\n\t\t" \
                            "tokenizer_name,\n\t\t" \
                            "word_index path."

                        assert "sent" in tokenizer_name, \
                            "%s model requires a tokenizer (spacy/nltk/bert/gpt2/roberta/xlnet)." % model_name
                        self.tokenizer_name = tokenizer_name

                        if model_name == "han":
                            assert "sent" in tokenizer_name, "HAN classifier requires " \
                                                             "sentence tokenizer. " \
                                                             "(nltk-sent or spacy-sent)"
                        else:
                            assert word_max_length, "Customized non transformer models require " \
                                                    "word_max_length specified."

                            self.word_max_length = word_max_length

    def __call__(self):
        config = self
        if self.model_name in transformer_names:
            if self.model_name == "bert":
                from transformers import BertConfig
                config = BertConfig
            elif self.model_name == "roberta":
                from transformers import RobertaConfig
                config = RobertaConfig
            elif self.model_name == "xlnet":
                from transformers import XLNetConfig
                config = XLNetConfig
            elif self.model_name == "gpt2":
                from transformers import GPT2Config
                config = GPT2Config

            if self.pretrained_model_name:
                config = config.from_pretrained(self.pretrained_model_name).from_dict(self.__dict__)
            elif self.pretrained_tokenizer_name:
                config = config().from_dict(self.__dict__)
            else:
                config = config().from_dict(self.__dict__)

        return config

    def to_dict(self):
        return self.__dict__

    def idx(self):
        return sha256(str(self.to_dict()).encode('utf-8')).hexdigest()


class TaskConfig:
    def __init__(self,
                 data_config: dict,
                 model_config: dict,
                 batch_size: int,
                 loss_func,
                 optimizer,
                 device: str = "cpu",
                 test=None,
                 epoch: int = 1,
                 freeze_emb: bool = True,
                 early_stop_epoch: int = 5,
                 test_only: bool = False,
                 retrain: bool = False,
                 random_seed: int = 0,
                 ):
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.device = device
        self.epoch = epoch
        self.test = test
        self.optimizer = optimizer
        self.freeze_emb = freeze_emb
        self.early_stop_epoch = early_stop_epoch
        self.random_seed = random_seed

        self.test_only = test_only
        self.retrain = retrain

        self.model_config = model_config
        self.data_config = data_config

    def __call__(self):
        self.model = ModelConfig(**self.model_config)
        self.data = DataConfig(**self.data_config)
        return self

    def to_dict(self):
        self.model_config = self.model.to_dict()
        self.data_config = self.data.to_dict()

        task = copy.deepcopy(self.__dict__)

        task['loss_func'] = str(task['loss_func'])
        task['optimizer'] = str(task['optimizer'])
        del task['batch_size']
        if "emb_path" in task["model_config"]:
            del task['model_config']['emb_path']
        if "word_max_length" in task['model_config']:
            del task['model_config']['word_max_length']
        try:
            del task['model']
            del task['data']
            del task['random_seed']
            return task
        except KeyError:
            return task

    def idx(self):
        return sha256(str(self.to_dict()).encode('utf-8')).hexdigest()