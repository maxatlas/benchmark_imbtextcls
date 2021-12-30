import torch

import config

from datetime import datetime
from tqdm import tqdm
from TaskDataset import split_tds, TaskDataset
from torch.utils.data import DataLoader, RandomSampler
from model_config import *
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from torch.nn.utils.rnn import pad_sequence


def collate_multi_label(batch):
    """
    input_ids & labels(could be string) -> padded input_ids and label_ids

    :param batch:
    :return:
    """
    text_ids, attention_mask, token_type_ids, label_ids = zip(*batch)

    text_ids = [torch.tensor(text_id) for text_id in text_ids]
    text_ids = pad_sequence(text_ids, batch_first=True, padding_value=0)

    attention_mask = [torch.tensor(mask) for mask in attention_mask]
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    limit = len(text_ids[0])

    token_type_ids = [[0] * limit]*len(text_ids)
    token_type_ids = torch.tensor(token_type_ids)

    label_ids = torch.tensor(label_ids, dtype=torch.float)

    return text_ids, attention_mask, token_type_ids, label_ids


def collate_single_label(batch):
    """
    input_ids & labels(could be string) -> padded input_ids and label_ids

    :param batch:
    :return:
    """
    text_ids, attention_mask, token_type_ids, label_list = zip(*batch)

    text_ids = [torch.tensor(text_id) for text_id in text_ids]
    text_ids = pad_sequence(text_ids, batch_first=True, padding_value=0)

    attention_mask = [torch.tensor(mask) for mask in attention_mask]
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    limit = len(text_ids[0])

    token_type_ids = [[0] * limit] * len(text_ids)
    token_type_ids = torch.tensor(token_type_ids)
    label_list = torch.tensor(label_list)

    return text_ids, attention_mask, token_type_ids, label_list


def train_per_ds(filename, model_name:str, batch_size:int, max_length:int, epoch:int, loss_func, test:int, hidden_size=768, device="cpu"):
    train_set, test_set, val_set = split_tds(filename, config.training_config['split_strategy'])
    train_set.data = train_set.data[:test]
    train_set.labels = train_set.labels[:test]
    train_set.set_label_ids() # labels -> label_ids

    test_set.labels = test_set.labels[:test]
    test_set.data = test_set.data[:test]

    print("\nLoading tokenizer ...")
    tokenizer = models[model_name]['tokenizer']
    tokenizer.model_max_length = max_length

    print("\nLoading model ...")
    model_init_config_dict = models[model_name]['config']().to_dict()
    model_init_config_dict['n_positions'] = max_length
    model_init_config_dict['cls_no'] = len(val_set.labels_meta.names)
    model_init_config_dict['vocab_size'] = len(tokenizer)
    model_init_config = models[model_name]['config'].from_dict(model_init_config_dict)

    model = models[model_name]['model'](config=model_init_config)

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
    for _ in range(epoch):
        for batch in tqdm(train_dl, desc="Iteration"):
            # for t in batch: t.to(device)
            text_ids, attention_mask, token_type_ids, label_ids = batch
            loss = model.batch_train(text_ids, attention_mask, token_type_ids, label_ids, loss_func=loss_func)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print("\nEvaluating ...")
    model.eval()
    for batch in tqdm(test_dl, desc="Iteration"):
        text_ids, attention_mask, token_type_ids, labels = batch

        res = model.batch_eval(text_ids, attention_mask, token_type_ids, labels, val_set.labels_meta.names)

    if not test: model.save_pretrained(save_directory="models/%s/%s" %(model_name, datetime.today()),
                          save_config=True, state_dict=model.state_dict())

    print(res)
    return res


if __name__ == "__main__":
    from losses import tversky_loss, dice_loss
    filename, model_name, batch_size, max_length, epoch, loss_func, test = "dataset/imdb.tds", "roberta", 2000, 1024, 1, BCEWithLogitsLoss(), 3
    # loss_func = dice_loss
    res = train_per_ds(filename, model_name, batch_size, max_length, epoch, loss_func, test)