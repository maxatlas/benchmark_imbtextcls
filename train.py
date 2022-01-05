import torch
import numpy as np

from task_config import TaskConfig

from datetime import datetime
from tqdm import tqdm
from TaskDataset import split_tds
from torch.utils.data import DataLoader
from model_config import *
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from torch.nn.utils.rnn import pad_sequence
from utils import pad_sequence_to_length


def seed_random(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def HAN_collate_batch(batch, pad_to_length, word_max_length, sent_max_length):
    text_ids, attention_mask, token_type_ids, label_ids = zip(*batch)
    text_ids = [[sent + [0] * (word_max_length - len(sent)) for sent in doc] for doc in text_ids]  # pad words
    text_ids = [doc + [[0] * word_max_length] * (sent_max_length - len(doc)) for doc in text_ids]  # pad sentences
    text_ids = [torch.tensor(text_id) for text_id in text_ids]

    return text_ids, attention_mask, token_type_ids, label_ids


def collate_batch(batch, pad_to_length, word_max_length, multi_label):
    """
    input_ids & labels(could be string) -> padded input_ids and label_ids

    :param multi_label:
    :param pad_to_length:
    :param batch:
    :return:
    """
    text_ids, attention_mask, token_type_ids, label_ids = zip(*batch)

    # limit to max length
    text_ids = [text_id[:word_max_length] for text_id in text_ids]
    attention_mask = [mask[:word_max_length] for mask in attention_mask]

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
    if type(test_set.labels[0]) is list:
        test_set.set_label_ids()

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
    model_config_dict['batch_size'] = batch_size
    model_config = models[model_name]['config']
    # TODO: clean this.
    for key, value in model_config_dict.items():
        try: model_config.__setattr__(key, value)
        except Exception: continue

    model = models[model_name]['model'](config=model_config)

    print("\nTokenizing ...")
    tknzed = tokenizer(train_set.data)
    train_set.data, train_set.attention_mask, train_set.token_type_ids = \
        tknzed.get('input_ids'), tknzed.get('attention_mask'), tknzed.get('token_type_ids')

    tknzed = tokenizer(test_set.data)
    test_set.data, test_set.attention_mask, test_set.token_type_ids = \
        tknzed.get('input_ids'), tknzed.get('attention_mask'), tknzed.get('token_type_ids')
    # model.to(device)

    collate_fn = HAN_collate_batch if task_config.model_name == "han" else collate_batch
    train_dl = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b,
                                        model_config.pad_to_length,
                                        model_config.word_max_length,
                                        True))
    test_dl = DataLoader(
        test_set, batch_size=batch_size,shuffle=True,
        collate_fn=lambda b: collate_fn(b,
                                        model_config.pad_to_length,
                                        model_config.word_max_length,
                                        task_config.multi_label))

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
        # batch = tuple(t.to("cuda:0") for t in batch)
        text_ids, attention_mask, token_type_ids, labels = batch

        res = model.batch_eval(text_ids, attention_mask, token_type_ids, labels, model_config.label_names)

    if not test: model.save_pretrained(save_directory="models/%s/%s" %(model_name, datetime.today()),
                                       save_config=True, state_dict=model.state_dict())

    print(res)
    return res


if __name__ == "__main__":
    from losses import tversky_loss, dice_loss
    seed_random(100)
    model_config_dict = {"hidden_dropout_prob": 0.1,
                         "emb_path": "models/emb_layer_glove",
                         "num_filters": 3,
                         "padding": 0,
                         "dilation": 1,
                         "max_length": 1024,
                         "filters": [2, 3, 4],
                         "stride": 1,
                         "word_max_length": 1024,
                         "sent_max_length": 50,
                         "pad_to_length": 1024,
                         "word_hidden_size":10,
                         "sent_hidden_size":10,
                         }

    conf_dict = {"filename": "dataset/ag_news.tds",
                 "emb_path": "models/emb_layer_glove",
                 "model_name": "han",
                 "batch_size": 100,
                 "max_length": 1024,
                 "epoch": 1,
                 "loss_func": BCEWithLogitsLoss(),
                 "device": "cpu",
                 "split_strategy": "uniform",
                 "test": 3,
                 "hidden_size": 10,
                 "multi_label": False}

    task_config = TaskConfig()

    model_names = ['bert', 'roberta', 'gpt', 'xlnet', 'lstm', 'cnn', 'rcnn', 'han']
    for model_name in model_names[:]:
        conf_dict['model_name'] = model_name
        task_config.from_dict(conf_dict)
        print(task_config.model_name)

        res = train_per_ds(task_config, model_config_dict)


