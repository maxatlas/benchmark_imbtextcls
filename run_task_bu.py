import dill
import os
import torch
import build_model
import build_dataset
import hashlib
import numpy as np
import json
import vars
import shutil
import vars

from tqdm import tqdm
from Config import (TaskConfig,
                    DataConfig, )
from torch.utils.data import DataLoader
from time import time
from task_utils import metrics_frame
from dataset_utils import get_max_lengths

shutil.rmtree(vars.cache_folder+"/results_old", ignore_errors=True)


def write_roc_list(debug_list: list, filename: str):
    file = open(filename, "a")
    file.writelines(debug_list)
    file.close()


def save_result(task: TaskConfig, results: dict, roc_list: list,
                cache=False, save_folder: str=None):
    idx = task.idx()

    folder = save_folder
    if not save_folder:
        folder = "_".join(task.data.huggingface_dataset_name) \
                 + "_balance_strategy_%s" % task.data.balance_strategy
        folder = "%s/%s" % (vars.results_folder) if not cache \
        else "%s/%s/%s" % (vars.cache_folder, "results_old", folder)

    os.makedirs(folder, exist_ok=True)

    filename = "%s/%s" % (folder, task.model.model_name)
    res = {
        idx: {
            "result": [results],
            "task": task.to_dict()
        }
    }

    roc_res = {
        idx: roc_list
    }

    try:
        results = dill.load(open(filename, "rb"))
        print(idx in results)
        if idx in results:
            results[idx]['result'].extend(res[idx]['result'])
        else:
            results.update(res)
        dill.dump(results, open(filename, "wb"))
        json.dump(results, open(filename + ".json", "w"))

        roc_results = dill.load(open(filename + ".roc", "rb"))
        roc_results.update(res)
        dill.dump(roc_results, open(filename + ".roc", "wb"))

    except FileNotFoundError:
        dill.dump(res, open(filename, "wb"))
        json.dump(res, open(filename + ".json", "w"))
        dill.dump(roc_res, open(filename + ".roc", "wb"))

    except KeyError:
        dill.dump(res, open(filename, "wb"))
        json.dump(res, open(filename + ".json", "w"))


def load_cache(config: dict):
    idx = hashlib.sha256(str(config).encode('utf-8')).hexdigest()
    sub_folder = "dataset" if "huggingface_dataset_name" in config else "model"

    filename = "%s/%s/%s/%s" % (vars.cache_folder, "exp", sub_folder, idx)
    try:
        return dill.load(open(filename, "rb"))
    except FileNotFoundError:
        return None


def cache(config: dict, data):
    idx = hashlib.sha256(str(config).encode('utf-8')).hexdigest()
    sub_folder = "dataset" if "huggingface_dataset_name" in config else "model"
    folder = "%s/%s/%s" % (vars.cache_folder, "exp", sub_folder)
    filename = "%s/%s" % (folder, idx)

    os.makedirs(folder, exist_ok=True)
    dill.dump(data, open(filename, 'wb'))


def main(task: TaskConfig, model_path=""):
    task.model_config["device"] = task.device
    task.data_config["test"] = task.test

    data_card = DataConfig(**task.data_config)

    print("Loading data ...")
    data = load_cache(task.data_config)

    if not data:
        data = build_dataset.main(data_card)
        cache(task.data_config, data)

    print("Loading model ...")
    if not model_path:
        model = load_cache(task.model_config)
    else:
        try:
            model = torch.load(model_path)
        except FileNotFoundError:
            model = None

    if type(data[0].label_feature) is list:
        task.model_config["num_labels"] = data[0].label_feature[0].num_classes
    else:
        task.model_config["num_labels"] = data[0].label_feature.num_classes

    word_max_length, sent_max_length = get_max_lengths(data[0].data)
    if not word_max_length:
        word_max_length = sent_max_length

    word_max_length = 512 if word_max_length > 512 else word_max_length

    if model_path:
        word_max_length = 512

    task.model_config["word_max_length"] = word_max_length
    model_card = task().model
    if not model:
        model = build_model.main(model_card)
        cache(model_card.to_dict(), model)
    if task.freeze_emb:
        model.freeze_emb()

    from model_utils import get_model_param_size
    print("model size: %i" % get_model_param_size(model))

    train_tds, test_tds, val_tds, split_info = data

    task.data.split_info = split_info
    train_dl = DataLoader(train_tds, batch_size=task.batch_size, shuffle=True)
    test_dl = DataLoader(test_tds, batch_size=task.batch_size, shuffle=True)
    # val_dl = DataLoader(val_tds, batch_size=task.batch_size, shuffle=True)

    optimizer = task.optimizer(model.parameters(), lr=model_card.lr)

    print("\t Training ...")
    clocks = 0

    acc_list = []
    valid_i = None

    if task.test_only:
        train_dl = DataLoader([])
        task.epoch = 1


    for i in range(task.epoch):
        print("Task running with: \n\t\t dataset %s" % task.data_config["huggingface_dataset_name"])
        print("\t\t model %s" % task.model_config['model_name'])
        print(model_card.to_dict())

        torch.cuda.empty_cache()

        probs_test, preds_test, labels_test = None, None, None

        print("\t epoch %s" % str(i))

        model.train()
        clock_start = time()
        for batch in tqdm(train_dl, desc="Iteration"):
            texts, labels = batch
            labels = torch.tensor(labels)
            label_feature = train_tds.label_feature
            loss = model.batch_train(texts, labels, label_feature.names, task.loss_func, train_tds.multi_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch %i finished." % i)
        clocks += time() - clock_start

        label_feature = test_tds.label_feature

        print("\t Evaluating ...")

        model.eval()
        for batch in tqdm(test_dl, desc="Iteration"):
            texts, labels = batch
            labels = labels.tolist()
            probs, preds, labels = model.batch_eval(texts, labels, label_feature.names, test_tds.multi_label)
            if preds_test is None and labels_test is None:
                probs_test, preds_test, labels_test = probs.cpu().numpy(), preds.cpu().numpy(), labels.cpu().numpy()
            else:
                probs_test = np.append(probs_test, preds.cpu().numpy(), axis=0)
                preds_test = np.append(preds_test, preds.cpu().numpy(), axis=0)
                labels_test = np.append(labels_test, labels.cpu().numpy(), axis=0)

        res = metrics_frame(probs_test, preds_test, labels_test, label_feature.names)

        print("\tAccuracy so far:")
        print(acc_list)
        print("Accuracy this epoch: %f" % res["Accuracy"])

        # If the accuracy is lower than half of the previous results_old ...
        if acc_list and res["Accuracy"] <= acc_list[-1]:
            threshold = len(acc_list) + task.early_stop_epoch
            print("###################%i %i#######################" % (i, threshold))
            if i > threshold:
                break
            continue

        valid_i = i
        valid_res = res
        acc_list.append(res["Accuracy"])
        print(res['Classification report'])

        if not task.test:
            save_result(task, res, [probs_test.tolist(), preds_test.tolist()], cache=True)
            print("\t Result cached ...")

    if not task.test:
        res = valid_res
        res['epochs'] = valid_i
        res['seconds_avg_epoch'] = clocks / (i + 1)

        if not model_path:
            save_result(task, res, [probs_test.tolist(), preds_test.tolist()])

            print("\t Result saved ...")


        train_folder = "%s/%s" % (vars.trained_model_folder, task.model.model_name) \
            if not model_path else "%s" % vars.trained_model_cur_folder
        os.makedirs(train_folder, exist_ok=True)

        file_name = task.idx() if not model_path else task.model.model_name
        torch.save(model, "%s/%s" % (train_folder, file_name))

        print("\t Model saved ...")

        return model
