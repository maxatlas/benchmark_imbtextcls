from Config import ModelConfig, DataConfig
from vars import datasets_meta
from torch.nn import BCEWithLogitsLoss  # CrossEntropyLoss, MSELoss,

import build_model
import build_dataset


def build(model_name, tokenizer_name, pretrained_model_name, pretrained_tokenizer_name):
    mc = ModelConfig(model_name, n_labels,
                     tokenizer_name=tokenizer_name,
                     pretrained_model_name=pretrained_model_name,
                     pretrained_tokenizer_name=pretrained_tokenizer_name,
                     word_max_length=50,
                     )
    return build_model.main(mc)


if __name__ == "__main__":
    dataset_i = 3
    test = 3
    dc = DataConfig(*datasets_meta[dataset_i].values())
    train_df, _, _ = build_dataset.main(dc)

    n_labels = train_df.label_feature.num_classes

    texts = list(train_df.data[datasets_meta[dataset_i]['text_fields'][0]][:test])
    labels = list(train_df.data[datasets_meta[dataset_i]['label_field']][:test])

    print("Scenario 1. Pretrained model.")
    model_name = "roberta"
    pretrained_model_name = "roberta-base"
    model = build(model_name, None, pretrained_model_name, None)

    model.batch_train(texts, labels, train_df.label_feature.names, BCEWithLogitsLoss())
    model.batch_eval(texts, labels, train_df.label_feature.names)


    print("Scenario 2. Transformer with pretrained tokenizer.")
    model_name = "roberta"
    pretrained_tokenizer_name = "roberta-base"
    model = build(model_name, None, None, pretrained_tokenizer_name)

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

    model.batch_train(texts, labels, train_df.label_feature.names, BCEWithLogitsLoss())
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
                     emb_path="params/emb_layer_glove",
                     word_index_path="params/word_index"
                     )
    model = build_model.main(mc)

    model.batch_train(texts, labels, train_df.label_feature.names, BCEWithLogitsLoss())
    model.batch_eval(texts, labels, train_df.label_feature.names)


