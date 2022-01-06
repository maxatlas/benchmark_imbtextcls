import torch
import numpy as np

from datetime import datetime
from tqdm import tqdm
from TaskDataset import split_tds
from torch.utils.data import DataLoader
from model_config import models, ModelConfig
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from utils import get_max_lengths


class TaskConfig(ModelConfig):
    def __init__(self):
        super(TaskConfig, self).__init__()


def seed_random(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def han_collate_batch(batch, word_max_length, sent_max_length, multi_label):
    text_ids, attention_mask, token_type_ids, label_ids = zip(*batch)
    text_ids = [[(sent + [0] * (word_max_length - len(sent)))
                 [:word_max_length] for sent in doc] for doc in text_ids]  # pad words
    text_ids = [(doc + [[0] * word_max_length] * (sent_max_length - len(doc)))
                [:sent_max_length] for doc in text_ids]  # pad sentences
    text_ids = torch.tensor(text_ids)

    label_ids = torch.tensor(label_ids, dtype=torch.float if multi_label else torch.long)

    return text_ids, attention_mask, token_type_ids, label_ids


def collate_batch(batch, word_max_length, sent_max_length, multi_label):
    """
    input_ids & labels(could be string) -> padded input_ids and label_ids

    :param multi_label:
    :param pad_to_length:
    :param batch:
    :return:
    """
    text_ids, attention_mask, token_type_ids, label_ids = zip(*batch)

    # limit to max length
    text_ids = [(doc + [0] * (word_max_length - len(doc)))
                [:word_max_length] for doc in text_ids]  # pad words
    attention_mask = [(mask + [0] * (word_max_length - len(mask)))
                      [:word_max_length] for mask in attention_mask]

    text_ids = torch.tensor(text_ids)
    attention_mask = torch.tensor(attention_mask)

    token_type_ids = [[0] * word_max_length] * len(text_ids)
    token_type_ids = torch.tensor(token_type_ids)

    label_ids = torch.tensor(label_ids, dtype=torch.float if multi_label else torch.long)

    return text_ids, attention_mask, token_type_ids, label_ids


def train_per_ds(task_config, model_config_d):
    test = task_config.test
    model_name = task_config.model_name
    batch_size = test if test < task_config.batch_size else task_config.batch_size

    train_set, test_set, val_set = split_tds(task_config.filename, task_config.split_strategy)

    word_max_length, sent_max_length = (get_max_lengths(train_set.data + test_set.data)[1], 1)\
                                           if task_config.filename.endswith(".tds") \
                                           else get_max_lengths(train_set.data + test_set.data)

    train_set.data = train_set.data[:test]
    train_set.labels = train_set.labels[:test]

    train_set.set_label_ids() # labels -> label_ids
    model_config_d['multi_label'] = False
    if type(test_set.labels[0]) is list:
        model_config_d['multi_label'] = True
        test_set.set_label_ids()

    test_set.labels = test_set.labels[:test]
    test_set.data = test_set.data[:test]

    print("\nLoading tokenizer ...")
    tokenizer = models[model_name]['tokenizer']
    tokenizer.model_max_length = word_max_length

    print("\nLoading model ...")
    model_config_dict = models[model_name]['config'].to_dict()
    model_config_dict.update(model_config_d)
    model_config_dict['n_positions'] = word_max_length
    model_config_dict['label_names'] = val_set.labels_meta.names
    model_config_dict['num_labels'] = len(val_set.labels_meta.names)
    model_config_dict['vocab_size'] = len(tokenizer)
    model_config_dict['max_position_embeddings'] = word_max_length
    model_config_dict['pad_token_id'] = 0
    model_config_dict['emb_path'] = task_config.emb_path
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

    collate_fn = han_collate_batch if model_name == "han" else collate_batch

    train_dl = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b,
                                        word_max_length,
                                        sent_max_length,
                                        True))
    test_dl = DataLoader(
        test_set, batch_size=batch_size,shuffle=True,
        collate_fn=lambda b: collate_fn(b,
                                        word_max_length,
                                        sent_max_length,
                                        model_config.multi_label))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    print("\nTraining ...")
    model.train()
    if test: print(train_set.data)
    for _ in range(task_config.epoch):
        for batch in tqdm(train_dl, desc="Iteration"):
            # batch = tuple(t.to("cuda:0") for t in batch)
            text_ids, attention_mask, token_type_ids, label_ids = batch
            if test: print(text_ids)
            loss = model.batch_train(text_ids, attention_mask, token_type_ids, label_ids,
                                     loss_func=task_config.loss_func)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print("\nEvaluating ...")
    model.eval()
    for batch in tqdm(test_dl, desc="Iteration"):
        # batch = tuple(t.to("cuda:0") for t in batch)
        text_ids, attention_mask, token_type_ids, labels = batch
        res = model.batch_eval(text_ids, attention_mask, token_type_ids, labels, model_config.label_names)

    if not test: model.save_pretrained(save_directory="models/%s/%s" % (model_name, datetime.today()),
                                       save_config=True, state_dict=model.state_dict())

    print(res)
    return res


if __name__ == "__main__":
    from losses import tversky_loss, dice_loss
    seed_random(100)
    model_dict = {"hidden_dropout_prob": 0.1,
                         "emb_path": "models/emb_layer_glove",
                         "num_filters": 3,
                         "padding": 0,
                         "dilation": 1,
                         "filters": [2, 3, 4],
                         "stride": 1,
                         # "hidden_size": 10,
                         "sent_hidden_size": 10,
                         "word_hidden_size": 10,
                         }

    task_dict = {"filename": "dataset/ag_news.tds",
                 "emb_path": "models/emb_layer_glove",
                 "model_name": "gpt2",
                 "batch_size": 100,
                 "epoch": 1,
                 "loss_func": BCEWithLogitsLoss(),
                 "device": "cpu",
                 "split_strategy": "uniform",
                 "test": 3,}

    task_config = TaskConfig()

    model_names = ['bert', 'roberta', 'gpt', 'xlnet', 'lstm', 'cnn', 'rcnn', 'lstmattn', 'han']
    for model_name in model_names[:]:
        if model_names.index(model_name) > 3: task_dict['filename'] = task_dict['filename'].replace(".tds", ".wi")

        task_dict['model_name'] = model_name
        task_config.from_dict(task_dict)
        print("\n"+model_name)

        res = train_per_ds(task_config, model_dict)


