"""
check max word length and sent length, configure model word max length accordingly properly
"""

from Config import ModelConfig, DataConfig

import build_model
import build_dataset
import torch

if __name__ == "__main__":
    from vars import datasets_meta
    from utils import save_transformer_emb, load_transformer_emb
    from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
    dataset_i = 1
    model_name = "xlnet"
    emb_path = ""

    mc = ModelConfig(model_name, 100, 2, 20, model_name, emb_layer_path=emb_path)
    model = build_model.main(mc)
    weight = save_transformer_emb(model, model_name)


    dc = DataConfig(*datasets_meta[dataset_i].values())
    train_df, _, _ = build_dataset.main(dc)

    texts = list(train_df.data[datasets_meta[dataset_i]['text_fields'][0]][:3])
    labels = list(train_df.data[datasets_meta[dataset_i]['label_field']][:3])
    labels = [int(i>0.5) for i in labels]

    loss = model.batch_train(texts, labels, train_df.label_feature.names, BCEWithLogitsLoss())
    # params = dict(model.named_parameters())
    # params.keys()