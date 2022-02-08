from Config import ModelConfig, DataConfig
from vars import datasets_meta
from torch.nn import BCEWithLogitsLoss , CrossEntropyLoss, MSELoss
from random_task import get_model_param_size
import build_model
import build_dataset
import vars
import torch
from task_utils import *


def build(model_name, tokenizer_name, pretrained_model_name, pretrained_tokenizer_name, n_labels):
    mc = ModelConfig(model_name, n_labels,
                     tokenizer_name=tokenizer_name,
                     pretrained_model_name=pretrained_model_name,
                     pretrained_tokenizer_name=pretrained_tokenizer_name,
                     word_max_length=50,
                     )
    return build_model.main(mc)


if __name__ == "__main__":
    dataset_i = 5
    test = 1
    dc = DataConfig(**datasets_meta[dataset_i])
    train_df, _, _, _ = build_dataset.main(dc)

    n_labels = train_df.label_feature.num_classes

    texts = list(train_df.data[:test])
    labels = list(train_df.labels[:test])

    labels = torch.tensor(labels)

    mc = ModelConfig("xlnet", n_labels,
                     pretrained_tokenizer_name="gpt2",
                     word_max_length=512,
                     emb_path="%sparams/emb_layer_gpt2" % vars.current,
                     n_layers=1,
                     qkv_size=50,
                     device="cpu")

    m = build_model.main(mc)
    m.freeze_emb()
    print(get_model_param_size(m))

    out = m.batch_train(texts, labels, train_df.label_feature.names,
                        loss_func=BCEWithLogitsLoss(), multi_label=False)
    print(out)

"""

    mc = ModelConfig(model_name, n_labels,
                     tokenizer_name=tokenizer,
                     pretrained_tokenizer_name="bert-base-uncased",
                     word_max_length=word_max_length,
                     emb_path="%sparams/emb_layer_bert" % vars.current,
                     n_layers=3)

    model = build_model.main(mc)

    out = model.batch_train(texts, labels, train_df.label_feature.names, loss_func=BCEWithLogitsLoss(),
                            multi_label=False)
    print(out)

    print("Scenario 1. Pretrained model.")
    model_name = "bert"
    pretrained_model_name = "bert-base-uncased"
    model = build(model_name, None, pretrained_model_name, None)

    model.batch_train(texts, labels, train_df.label_feature.names, BCEWithLogitsLoss())
    model.batch_eval(texts, labels, train_df.label_feature.names)


    print("Scenario 2. Transformer with pretrained tokenizer.")
    model_name = "gpt2"
    pretrained_tokenizer_name = "bert-base-uncased"
    mc = ModelConfig(
        model_name,
        n_labels,
        pretrained_tokenizer_name=pretrained_tokenizer_name,
        n_layers=1,
        hidden_size=50,
        n_heads=1,
        device="cpu"
    )
    model = build_model.main(mc)
    model.batch_train(texts, labels, train_df.label_feature.names, BCEWithLogitsLoss())
    model.batch_eval(texts, labels, train_df.label_feature.names)


    print("Scenario 3. Transformer with customized tokenizer.")
    model_name = "xlnet"
    tokenizer_name = "xlnet"
    try:
        build(model_name, tokenizer_name, None, None)
    except Exception as e:
        print(e)


    print("Scenario 4. Model with pretrained tokenizer.")
    model_name = "lstmattn"
    pretrained_tokenizer_name = "xlnet-base-cased"
    model = build(model_name, None, None, pretrained_tokenizer_name)

    model.batch_train(texts, labels, train_df.label_feature.names, BCEWithLogitsLoss(), dc.multi_label)
    model.batch_eval(texts, labels, train_df.label_feature.names)


    print("Scenario 5. Model with customized transformer tokenizer.")
    model_name = "lstmattn"
    tokenizer_name = "roberta"
    try:
        build(model_name, tokenizer_name, None, None)
    except Exception as e:
        print(e)


    print("Scenario 6. Model with customized tokenizer.")
    model_name = "cnn"
    tokenizer_name = "nltk"

    mc = ModelConfig(model_name, n_labels,
                     tokenizer_name=tokenizer_name,
                     word_max_length=50,
                     emb_path="%s/emb_layer_glove" % vars.parameter_folder,
                     word_index_path="%s/word_index" % vars.parameter_folder,
                     )
    model = build_model.main(mc)

    model.batch_train(texts, labels, train_df.label_feature.names, BCEWithLogitsLoss())
    model.batch_eval(texts, labels, train_df.label_feature.names)


"""