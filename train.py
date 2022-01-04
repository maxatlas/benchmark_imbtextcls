import torch

from config import TaskConfig

from datetime import datetime
from tqdm import tqdm
from TaskDataset import split_tds, TaskDataset
from torch.utils.data import DataLoader, RandomSampler
from model_config import *
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from torch.nn.utils.rnn import pad_sequence
from utils import pad_sequence_to_length

torch.manual_seed(0)


def collate_batch(batch, pad_to_length, multi_label):
    """
    input_ids & labels(could be string) -> padded input_ids and label_ids

    :param batch:
    :return:
    """
    text_ids, attention_mask, token_type_ids, label_ids = zip(*batch)

    text_ids = [torch.tensor(text_id) for text_id in text_ids]
    text_ids = pad_sequence_to_length(text_ids, pad_to_length, batch_first=True, padding_value=0) \
        if pad_to_length \
        else pad_sequence(text_ids, batch_first=True, padding_value=0)

    attention_mask = [torch.tensor(mask) for mask in attention_mask]
    attention_mask = pad_sequence_to_length(attention_mask, pad_to_length, batch_first=True, padding_value=0) \
        if pad_to_length \
        else pad_sequence(attention_mask, batch_first=True, padding_value=0)

    limit = len(text_ids[0])

    token_type_ids = [[0] * limit] * len(text_ids)
    token_type_ids = torch.tensor(token_type_ids)

    label_ids = torch.tensor(label_ids, dtype=torch.float if multi_label else torch.long)

    return text_ids, attention_mask, token_type_ids, label_ids


def train_per_ds(task_config, model_config_d):
    filename = task_config.filename
    test = task_config.test
    model_name = task_config.model_name
    max_length = task_config.max_length
    batch_size = task_config.batch_size
    epoch = task_config.epoch
    loss_func = task_config.loss_func
    split_strategy = task_config.split_strategy
    emb_path = task_config.emb_path

    train_set, test_set, val_set = split_tds(filename, split_strategy)
    train_set.data = train_set.data[:test]
    train_set.labels = train_set.labels[:test]
    train_set.set_label_ids() # labels -> label_ids

    test_set.labels = test_set.labels[:test]
    test_set.data = test_set.data[:test]

    print("\nLoading tokenizer ...")
    tokenizer = models[model_name]['tokenizer']
    tokenizer.model_max_length = max_length

    print("\nLoading model ...")
    model_config_dict = models[model_name]['config'].to_dict()
    model_config_dict.update(model_config_d)
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

    train_dl = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_batch(b,
                                           model_config.pad_to_length,
                                           True))
    test_dl = DataLoader(test_set, batch_size=batch_size,
                         shuffle=True, collate_fn=lambda b: collate_batch(b,
                                                                          model_config.pad_to_length,
                                                                          model_config.multi_label))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    print("\nTraining ...")
    model.train()
    for _ in range(epoch):
        for batch in tqdm(train_dl, desc="Iteration"):
            # batch = tuple(t.to("cuda:0") for t in batch)
            text_ids, attention_mask, token_type_ids, label_ids = batch
            loss = model.batch_train(text_ids, attention_mask, token_type_ids, label_ids, loss_func=loss_func)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print("\nEvaluating ...")
    model.eval()
    for batch in tqdm(test_dl, desc="Iteration"):
        text_ids, attention_mask, token_type_ids, labels = batch

        res = model.batch_eval(text_ids, attention_mask, token_type_ids, labels, model_config.label_names)

    if not test: model.save_pretrained(save_directory="models/%s/%s" %(model_name, datetime.today()),
                          save_config=True, state_dict=model.state_dict())

    print(res)
    return res


if __name__ == "__main__":
    from losses import tversky_loss, dice_loss

    model_config_dict = {"hidden_dropout_prob": 0.1,
                         "emb_path": "models/emb_layer_glove",
                         "num_filters": 3,
                         "padding": 0,
                         "dilation": 1,
                         "max_length": 1024,
                         "filters": [2, 3, 4],
                         "stride": 1,
                         "pad_to_length": 1024,
                         "multi_label": False}

    conf_dict = {"filename": "dataset/imdb.wi",
                 "emb_path": "models/emb_layer_glove",
                 "model_name": "cnn",
                 "batch_size": 100,
                 "max_length": 1024,
                 "epoch": 1,
                 "loss_func": BCEWithLogitsLoss(),
                 "device": "cpu",
                 "split_strategy": "uniform",
                 "test": 3,
                 "hidden_size": 10}

    task_config = TaskConfig()
    task_config.from_dict(conf_dict)

    # loss_func = dice_loss
    res = train_per_ds(task_config, model_config_dict)
