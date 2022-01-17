import dill
import os
import torch
import build_model
import build_dataset
import hashlib
import numpy as np
import json
import vars
from vars import (cache_folder,
                  results_folder)
from tqdm import tqdm
from Config import (TaskConfig,
                    DataConfig,)
from torch.utils.data import DataLoader


from time import time
from task_utils import metrics_frame
from dataset_utils import get_max_lengths

torch.cuda.empty_cache()


def save_result(task: TaskConfig, results: dict):
    """
    Under folder of dataset name and balance strategy
    filename is model name _task.idx()
    append, if filename exists
    :param task:
    :param results:
    :return:
    """

    folder = "_".join(task.data.huggingface_dataset_name)\
                  + "_balance_strategy_%s" % task.data.balance_strategy
    folder = "%s/%s" % (results_folder, folder)
    try:
        os.listdir(folder)
    except FileNotFoundError:
        os.makedirs(folder, exist_ok=True)

    filename = "%s/%s" % (folder, task.model_config["model_name"])
    res = {
        task.idx(): {
            "model": task.model_config,
            "result": results,
            "task": task.to_dict()
        }
    }

    try:
        ress = dill.load(open(filename, "rb"))
        ress.update(res)

        dill.dump(ress, open(filename, "wb"))
        json.dump(str(res), open(filename+".json", "a"))
    except FileNotFoundError:
        dill.dump(res, open(filename, "wb"))
        json.dump(str(res), open(filename+".json", "w"))
        

def load_cache(config: dict):
    idx = hashlib.sha256(str(config).encode('utf-8')).hexdigest()

    filename = "%s/%s" % (cache_folder, idx)
    try:
        return dill.load(open(filename, "rb"))
    except FileNotFoundError:
        return None


def cache(config: dict, data):
    idx = hashlib.sha256(str(config).encode('utf-8')).hexdigest()

    filename = "%s/%s" % (cache_folder, idx)
    try:
        dill.dump(data, open(filename, 'wb'))
    except FileNotFoundError:
        os.makedirs(cache_folder, exist_ok=True)


def main(task: TaskConfig):
    print("Task running with: \n\t\t dataset %s" % task.data_config["huggingface_dataset_name"])
    print("\t\t model %s" % task.model_config['model_name'])

    task.model_config["device"] = task.device
    task.data_config["test"] = task.test
    print(str(task.model_config))

    data_card = DataConfig(**task.data_config)

    print("Loading data ...")
    data = load_cache(task.data_config)

    if not data:
        data = build_dataset.main(data_card)
        if not task.test:
            cache(task.data_config, data)
    print("Loading model ...")
    model = load_cache(task.model_config)

    if type(data[0].label_feature) is list:
        task.model_config["num_labels"] = data[0].label_feature[0].num_classes
    else:
        task.model_config["num_labels"] = data[0].label_feature.num_classes

    word_max_length, sent_max_length = get_max_lengths(data[0].data)
    if not word_max_length:
        word_max_length = sent_max_length
    task.model_config["word_max_length"] = word_max_length
    model_card = task().model

    if not model:
        model = build_model.main(model_card)
        cache(task.model_config, model)

    train_tds, test_tds, val_tds, split_info = data

    train_dl = DataLoader(train_tds, batch_size=task.batch_size, shuffle=True)
    test_dl = DataLoader(test_tds, batch_size=task.batch_size, shuffle=True)
    # val_dl = DataLoader(val_tds, batch_size=task.batch_size, shuffle=True)

    optimizer = task.optimizer(model.parameters(), lr=model_card.lr)

    print("\t Training ...")

    clocks = 0

    preds_eval, labels_eval = None, None

    print(task.epoch)
    for i in range(task.epoch):
        print("\t epoch %s" % str(i))

        model.train()
        clock_start = time()
        for batch in tqdm(train_dl, desc="Iteration"):
            texts, labels = batch
            labels = torch.tensor(labels)
            label_feature = train_tds.label_feature[0] if train_tds.multi_label else train_tds.label_feature
            loss = model.batch_train(texts, labels, label_feature.names, task.loss_func, train_tds.multi_label)
            if model_card.model_name == "han":
                model._init_hidden_state(len(texts))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch %i finished." % i)
        clocks += time() - clock_start

        print("\t Evaluating ...")
        model.eval()
        for batch in tqdm(test_dl, desc="Iteration"):
            texts, labels = batch
            labels = labels.tolist()
            label_feature = test_tds.label_feature[0] if test_tds.multi_label else test_tds.label_feature
            if model_card.model_name == "han":
                model._init_hidden_state(len(texts))
            preds, labels = model.batch_eval(texts, labels, label_feature.names, test_tds.multi_label)
            if preds_eval is None and labels_eval is None:
                preds_eval, labels_eval = preds.cpu().numpy(), labels.cpu().numpy()
            else:
                preds_eval = np.append(preds_eval, preds.cpu().numpy(), axis=0)
                labels_eval = np.append(labels_eval, labels.cpu().numpy(), axis=0)

        res = metrics_frame(preds_eval, labels_eval,
                            label_feature.names)

        res['seconds_avg_epoch'] = clocks / (i + 1)
        res['split_info'] = split_info
        print(res['Classification report'])

        if not task.test:
            save_result(task, res)
            print("\t Result saved ...")

            train_folder = "%s/%s" % (vars.trained_model_folder,
                                      task.model.model_name)
            os.makedirs(train_folder, exist_ok=True)

            torch.save(model, "%s/%s" % (train_folder,
                                         task.idx()))
            print("\t Model saved ...")



