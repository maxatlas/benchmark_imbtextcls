import dill

import torch
import build_model
import build_dataset

from tqdm import tqdm
from Config import (TaskConfig)
from torch.utils.data import DataLoader

from datetime import datetime
from time import time
from utils import metrics_frame


def load_cache(card):
    filename = ".job_cache/%s" % card.idx()
    try:
        return dill.load(open(filename, "rb"))
    except FileNotFoundError:
        return None


def cache(card, data):
    filename = ".job_cache/%s" % card.idx()
    dill.dump(data, open(filename, 'wb'))


def main(task: TaskConfig):
    print("Task running with: \n\t\t dataset %s" % task.data.huggingface_dataset_name)
    print("\t\t model %s" % task.model.model_name)

    model_card = task.model
    data_card = task.data

    model_card.device = task.device
    data_card.test = task.test
    print(str(task.model.to_dict()))

    model = load_cache(model_card)
    data = load_cache(data_card)

    if not model:
        model = build_model.main(model_card)
        cache(model_card, model)
    if not data:
        data = build_dataset.main(data_card)
        cache(data_card, data)

    train_tds, test_tds, val_tds = data
    train_dl = DataLoader(train_tds, batch_size=task.batch_size, shuffle=True)
    test_dl = DataLoader(test_tds, batch_size=task.batch_size, shuffle=True)
    # val_dl = DataLoader(val_tds, batch_size=task.batch_size, shuffle=True)

    optimizer = task.optimizer(model.parameters(), lr=task.model.lr)

    model.train()
    print("\t Training ...")

    clocks = 0

    for i in range(task.epoch):
        print("\t epoch %s" % str(i))
        clock_start = time()
        for batch in tqdm(train_dl, desc="Iteration"):
            texts, labels = batch
            labels = labels.tolist()
            loss = model.batch_train(texts, labels, train_tds.label_feature.names, task.loss_func)
            if task.model.model_name == "han":
                model._init_hidden_state(len(texts))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch %i finished." %i)
        clocks += time() - clock_start


    print("\t Evaluating ...")
    model.eval()
    preds_eval, labels_eval = [], []

    for batch in tqdm(test_dl, desc="Iteration"):
        texts, labels = batch
        labels = labels.tolist()

        if task.model.model_name == "han":
            model._init_hidden_state(len(texts))
        preds = model.batch_eval(texts, labels, train_tds.label_feature.names)

        preds_eval.extend(preds)
        labels_eval.extend(labels)

    preds_eval = torch.tensor(preds_eval).cpu().numpy()
    labels_eval = torch.tensor(labels_eval).cpu().numpy()

    res = metrics_frame(preds_eval, labels_eval, train_tds.label_feature.names)
    res['seconds_avg_epoch'] = clocks/task.epoch
    if not task.test:
        model.save_pretrained(save_directory="models/%s/%s" % (task.model.model_name,
                                                                             datetime.today()),
                                            save_config=True, state_dict=model.state_dict())
    return res
