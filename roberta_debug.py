from TaskDataset import split_tds
from model_config import models
from torch.utils.data import DataLoader
from train import collate_batch, han_collate_batch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import torch

from utils import get_max_lengths

if __name__ == "__main__":
    # torch.manual_seed(66)
    test=3
    filename="dataset/ag_news.wi"
    max_length=1024
    model_name="han"
    batch_size= test if test else 100
    split_strategy="uniform"
    emb_path="models/emb_layer_glove"

    c = {"hidden_dropout_prob": 0.1,
         "emb_path": "models/emb_layer_glove",
         "num_filters": 3,
         "padding": 0,
         "dilation": 1,
         "max_length": 1024,
         "filters": [2, 3, 4],
         "stride": 1,
         "pad_to_length": 0,
         "sent_max_length": 50,
         "multi_label": False,
         "word_hidden_size": 10,
         "sent_hidden_size": 10,
         }

    train_set, test_set, val_set = split_tds(filename, split_strategy)

    word_max_length, sent_max_length = get_max_lengths(train_set.data + test_set.data)
    if word_max_length < max_length * 0.8: max_length = word_max_length

    train_set.data = train_set.data[:test]
    train_set.labels = train_set.labels[:test]
    train_set.set_label_ids()  # labels -> label_ids

    test_set.labels = test_set.labels[:test]
    test_set.data = test_set.data[:test]

    print("\nLoading tokenizer ...")
    tokenizer = models[model_name]['tokenizer']
    tokenizer.model_max_length = max_length

    print("\nLoading model ...")
    model_config_dict = models[model_name]['config'].to_dict()
    model_config_dict.update(c)
    model_config_dict['n_positions'] = max_length
    model_config_dict['label_names'] = val_set.labels_meta.names
    model_config_dict['num_labels'] = val_set.labels_meta.num_classes
    model_config_dict['vocab_size'] = len(tokenizer)
    model_config_dict['max_position_embeddings'] = max_length
    model_config_dict['pad_token_id'] = 0
    model_config_dict['emb_path'] = emb_path
    model_config_dict['pack_to_max'] = 1
    model_config_dict['batch_size'] = batch_size
    model_config = models[model_name]['config'].from_dict(model_config_dict)

    print("\nTokenizing ...")
    tknzed = tokenizer(train_set.data)
    train_set.data, train_set.attention_mask, train_set.token_type_ids = \
        tknzed.get('input_ids'), tknzed.get('attention_mask'), tknzed.get('token_type_ids')

    tknzed = tokenizer(test_set.data)
    test_set.data, test_set.attention_mask, test_set.token_type_ids = \
        tknzed.get('input_ids'), tknzed.get('attention_mask'), tknzed.get('token_type_ids')

    collate_fn = han_collate_batch if model_name == "han" else collate_batch
    train_dl = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b,
                                        model_config.pad_to_length,
                                        max_length,
                                        sent_max_length,
                                        True))
    test_dl = DataLoader(
        test_set, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b,
                                        model_config.pad_to_length,
                                        max_length,
                                        sent_max_length,
                                        model_config.multi_label))

    batch = next(iter(train_dl))

    text_ids, attention_mask, token_type_ids, label_ids = batch

    model = models[model_name]['model'](config=model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    print("\nTraining ...")
    model.train()

    # batch = tuple(t.to("cuda:0") for t in batch)
    out = model.forward(text_ids)
    logits = model.batch_train(text_ids, attention_mask, token_type_ids, label_ids, loss_func=BCEWithLogitsLoss())

    print("\nEvaluating ...")
    batch = next(iter(test_dl))
    text_ids, attention_mask, token_type_ids, label_ids = batch
    res = model.batch_eval(text_ids, attention_mask, token_type_ids, label_ids, label_names=val_set.labels_meta.names)
    print(res)