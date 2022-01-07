import hashlib
import dill


class DataConfig:
    def __init__(self,
                 huggingface_dataset_name,
                 label_field: str,
                 text_fields: list,
                 cls_ratio_to_imb: float,
                 sample_ratio_to_imb: float,
                 split_ratio="0.75/0.20/0.05",
                 train_set_transform=None,
                 threshold=0.6, tolerance=0.3,):
        assert not huggingface_dataset_name or type(huggingface_dataset_name) is list or tuple, \
            "huggingface_dataset_name wrongly formatted. A valid example: (glue, sst) or [glue, sst]"
        assert (type(split_ratio) is str and all([float(e) for e in split_ratio.split("/")])), \
            "Wrong format for split_ratio. Should be 'train/test/val'. A valid example: '0.75/0.2/0.05'"
        assert train_set_transform in [None, "undersample"], \
            "Train set distribution could be None or 'undersample'. Implement if need more."

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

        self.train_set_dist = train_set_transform

    def to_dict(self):
        return self.__dict__

    def save(self, folder="Jobs"):
        info = str(self.to_dict())

        filename = hashlib.sha256(info.encode('utf-8')).hexdigest()
        dill.dump(self, open("%s/%s" % (folder, filename), 'wb'))

        print("DataConfig object saved.")
        return filename


class ModelConfig:
    def __init__(self, model_name,
                 word_max_length,
                 n_labels,
                 tokenizer_name=None,
                 emb_path=None,
                 emb_layer_path=None,
                 hidden_size=None,
                 n_layers=None,
                 activation_function=None,
                 dropout=0.1,
                 padding=0,
                 dilation=1,
                 stride=1,
                 filters=(2, 3, 4, 5)):

        self.model_name = model_name.lower()
        assert self.model_name in ["bert", "xlnet", "roberta", "gpt2", "lstm",
                                   "lstmattn", "cnn", "rcnn", "han", "mlp"], \
            "Model name should be any of the following:" \
            "\n\t\tBert\n\t\tXLNet" \
            "\n\t\tRoBERTa\n\t\tGPT2\n\t\tLSTM" \
            "\n\t\tlstmattn (LSTM with attention) " \
            "\n\t\tCNN\n\t\tRCNN\n\t\tHAN\n\t\tMLP"

        self.tokenizer = None
        self.n_labels = n_labels

        self.tokenizer_name = tokenizer_name
        self.emb_path = emb_path

        if model_name == "bert" or "roberta":
            self.vocab_size = 30_522 if model_name == "bert" else 50_265
            self.max_position_embeddings = word_max_length

        if model_name == "gpt2":
            self.vocab_size = 50_257
            self.n_positions = word_max_length

        if model_name == "xlnet":
            self.vocab_size = 32_000

        else:
            assert tokenizer_name in ["spacy", "nltk", "bert", "gpt2"], \
                "%s model requires a tokenizer (spacy/nltk)." % model_name
            assert emb_path and tokenizer_name not in ["bert", "gpt2"], \
                "%s model requires an embedder (glove/fasttext/word2vec)."
            assert emb_layer_path, "%s model requires embedding layers."

            self.emb_d = 300
            self.dropout = dropout
            self.word_max_length = word_max_length
            self.activation = activation_function
            self.emb_layer_path = emb_layer_path

            if tokenizer_name in ["bert", "gpt2"]:
                self.emb_d = 768

            if model_name in ["lstm", "lstmattn", "rcnn", "mlp", "han"]:
                assert hidden_size, "%s model requires hidden_size attribute." % model_name
                self.hidden_size = hidden_size
                self.n_layers = n_layers if n_layers else 1

            elif model_name is "cnn":
                self.padding = padding
                self.dilution = dilation
                self.stride = stride
                self.filters = filters

    def __call__(self):
        config = self
        if self.model_name == "bert":
            from transformers import BertConfig
            config = BertConfig.from_pretrained("bert-base-uncased").from_dict(self.__dict__)
        elif self.model_name == "roberta":
            from transformers import RobertaConfig
            config = RobertaConfig.from_pretrained("roberta-base").from_dict(self.__dict__)
        elif self.model_name == "xlnet":
            from transformers import XLNetConfig
            config = XLNetConfig.from_pretrained("xlnet-large-cased").from_dict(self.__dict__)
        elif self.model_name == "gpt2":
            from transformers import GPT2Config
            config = GPT2Config.from_pretrained("gpt2").from_dict(self.__dict__)

        return config

    def to_dict(self):
        return self.__dict__

    def save(self, folder="Jobs"):
        info = str(self.to_dict())

        filename = hashlib.sha256(info.encode('utf-8')).hexdigest()
        dill.dump(self, open("%s/%s" % (folder, filename), 'wb'))

        print("ModelConfig object saved.")
        return filename


class TaskConfig:
    def __init__(self, model_config_dict: dict,
                 data_config_dict: dict,
                 batch_size: int,
                 loss_func,
                 device: str = "cpu",
                 test: int = 0,
                 epoch: int = 1,
                 ):

        self.batch_size = batch_size
        self.loss_func = loss_func
        self.device = device
        self.epoch = epoch
        self.test = test

        self.model_config_dict = model_config_dict
        self.data_config_dict = data_config_dict
        
        self.model_config = None 
        self.data_config = None

        self.__post_init__()
        
    def __post_init__(self):
        self.model_config = ModelConfig(**self.model_config_dict)()
        self.data_config = DataConfig(**self.data_config_dict)

    def to_dict(self):
        return self.__dict__

    def save(self, folder="Jobs"):
        info = str(self.to_dict())

        filename = hashlib.sha256(info.encode('utf-8')).hexdigest()
        dill.dump(self, open("%s/%s" % (folder, filename), 'wb'))

        print("TaskConfig object saved.")
        return filename


if __name__ == "__main__":
    mc = {"model_name": "gpt", "max_word_length": 1024, "n_labels": 2}
    dc = {}
    tc = TaskConfig(mc, dc, 3, print)