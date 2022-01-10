"""
check max word length and sent length, configure model word max length accordingly properly
"""

from Config import ModelConfig, DataConfig

import build_model
import build_dataset




if __name__ == "__main__":
    from vars import datasets_meta
    from torch.nn import BCEWithLogitsLoss  # CrossEntropyLoss, MSELoss,
    dataset_i = 3
    model_name = "roberta"
    pretrained_model_name = "roberta-base"
    emb_path = ""
    max_length = 0
    test = None

    dc = DataConfig(*datasets_meta[dataset_i].values(), test=test)
    train_df, _, _ = build_dataset.main(dc)

    n_labels = train_df.label_feature.num_classes

    mc = ModelConfig(model_name, n_labels,
                     pretrained_model_name=pretrained_model_name)
                     # "bert",
                     # pretrained_tokenizer_name="bert-base-uncased",
                     # emb_path="parameters/emb_layer_bert",
                     # hidden_size=20,
                     # word_max_length=100)

    model = build_model.main(mc)
    # weight = build_model.save_transformer_emb(model, model_name)

    texts = list(train_df.data[datasets_meta[dataset_i]['text_fields'][0]][:3])
    labels = list(train_df.data[datasets_meta[dataset_i]['label_field']][:3])

    loss = model.batch_train(texts, labels, train_df.label_feature.names, BCEWithLogitsLoss())
    res = model.batch_eval(texts, labels, train_df.label_feature.names)

    print(loss)
    print(res)
    # params = dict(model.named_parameters())
    # params.keys()
