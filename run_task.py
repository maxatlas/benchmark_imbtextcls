import dill
import os
import torch
import build_model
import build_dataset
import hashlib

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
    except FileNotFoundError:
        dill.dump(res, open(filename, "wb"))


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
        cache(task.data_config, data)

    print("Loading model ...")
    model = load_cache(task.model_config)

    task.model_config["num_labels"] = data[0].label_feature.num_classes
    word_max_length, sent_max_length = get_max_lengths(data[0].data)
    if not word_max_length:
        word_max_length = sent_max_length
    task.model_config["word_max_length"] = word_max_length
    model_card = task().model

    if not model:
        model = build_model.main(model_card)
        cache(task.model_config, model)

    train_tds, test_tds, val_tds = data
    train_dl = DataLoader(train_tds, batch_size=task.batch_size, shuffle=True)
    test_dl = DataLoader(test_tds, batch_size=task.batch_size, shuffle=True)
    # val_dl = DataLoader(val_tds, batch_size=task.batch_size, shuffle=True)

    optimizer = task.optimizer(model.parameters(), lr=model_card.lr)

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
            if model_card.model_name == "han":
                model._init_hidden_state(len(texts))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch %i finished." % i)
        clocks += time() - clock_start


    print("\t Evaluating ...")
    model.eval()
    preds_eval, labels_eval = [], []

    for batch in tqdm(test_dl, desc="Iteration"):
        texts, labels = batch
        labels = labels.tolist()
        if model_card.model_name == "han":
            model._init_hidden_state(len(texts))
        preds = model.batch_eval(texts, labels, train_tds.label_feature.names)
        preds_eval.extend(preds)
        labels_eval.extend(labels)

    preds_eval = torch.tensor(preds_eval).cpu().numpy()
    labels_eval = torch.tensor(labels_eval).cpu().numpy()

    res = metrics_frame(preds_eval, labels_eval, train_tds.label_feature.names)
    res['seconds_avg_epoch'] = clocks/task.epoch

    if not task.test:
        save_result(task, res)
        print("\t Result saved ...")

        train_folder = "%s/%s" % (vars.trained_model_folder,
                                        task.model.model_name)

        os.makedirs(train_folder, exist_ok=True)

        torch.save(model, "%s/%s" % (train_folder,
                                     task.idx()))
        print("\t Model saved ...")

    print(res)
    return res
