import classifiers.BERT as BERT
import classifiers.GPT2 as GPT2
import classifiers.Roberta as Roberta
import build_dataset

from vars import datasets_meta

from Config import *

if __name__ == "__main__":
    dataset_i = 10
    test = 3
    dc = DataConfig(**datasets_meta[dataset_i])
    train_df, _, _, _ = build_dataset.main(dc)

    n_labels = train_df.label_feature.num_classes

    texts = list(train_df.data[:test])
    labels = list(train_df.labels[:test])

    # mc = ModelConfig(model_name="Roberta",
    #                  pretrained_model_name="roberta-base",
    #                  num_labels=2)
    # model = Roberta.Model.from_pretrained("roberta-base")
    from transformers import GPT2Config, GPT2Tokenizer
    bc = GPT2Config()
    bc.disable_output = True
    bc.disable_intermediate = True
    bc.disable_selfoutput = True
    bc.emb_path = "params/emb_layer_bert"
    bc.cls_hidden_size = 50
    bc.num_labels = 2
    bc.num_hidden_layers = 2
    model = GPT2.Model(bc)
    model.forward(texts)
