from TaskDataset import split_tds
from model_config import models
from torch.utils.data import DataLoader
from train import collate_multi_label, collate_single_label
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import torch
import config

if __name__ == "__main__":
    # torch.manual_seed(66)
    test=3
    filename="dataset/imdb.wi"
    max_length=1024
    model_name="lstm"
    batch_size=100
    split_strategy="uniform"
    emb_path="models/emb_layer_glove"

    train_set, test_set, val_set = split_tds(filename, split_strategy)
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
    model_config_dict['n_positions'] = max_length
    model_config_dict['label_names'] = val_set.labels_meta.names
    model_config_dict['num_labels'] = len(val_set.labels_meta.names)
    model_config_dict['vocab_size'] = len(tokenizer)
    model_config_dict['max_position_embeddings'] = max_length
    model_config_dict['pad_token_id'] = 0
    model_config_dict['emb_path'] = emb_path
    model_config = models[model_name]['config'].from_dict(model_config_dict)

    model = models[model_name]['model'](config=model_config)

    print("\nTokenizing ...")
    tknzed = tokenizer(train_set.data)
    train_set.data, train_set.attention_mask, train_set.token_type_ids = \
        tknzed.get('input_ids'), tknzed.get('attention_mask'), tknzed.get('token_type_ids')

    tknzed = tokenizer(test_set.data)
    test_set.data, test_set.attention_mask, test_set.token_type_ids = \
        tknzed.get('input_ids'), tknzed.get('attention_mask'), tknzed.get('token_type_ids')

    # Limit by max length
    train_set.cut_samples(max_length)
    test_set.cut_samples(max_length)
    # model.to(device)

    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_multi_label)
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_single_label)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    print("\nTraining ...")
    model.train()

    batch = next(iter(train_dl))
    # batch = tuple(t.to("cuda:0") for t in batch)
    text_ids, attention_mask, token_type_ids, label_ids = batch

    logits = model.batch_train(text_ids, attention_mask, token_type_ids, label_ids, loss_func=BCEWithLogitsLoss())

    print("\nEvaluating ...")
    batch = next(iter(test_dl))
    text_ids, attention_mask, token_type_ids, label_ids = batch
    res = model.batch_eval(text_ids, attention_mask, token_type_ids, label_ids, label_names=val_set.labels_meta.names)
    print(res)