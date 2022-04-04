import dill
import os
import torch
import build_model
import build_dataset
import hashlib
import numpy as np
import json
import shutil
import vars
import copy

from tqdm import tqdm
from Config import (TaskConfig,
                    DataConfig, )
from torch.utils.data import DataLoader
from time import time
from task_utils import metrics_frame, set_random_seed
from dataset_utils import get_max_lengths

from pprint import pprint

shutil.rmtree(vars.cache_folder+"/results", ignore_errors=True)


def save_result(task: TaskConfig, results: dict, roc_list: list,
                cache=False, save_folder: str=None):
    idx = task.idx()

    folder = save_folder
    if not save_folder:
        folder = "_".join(task.data.huggingface_dataset_name)
        folder = "%s/%s" % (vars.results_folder, folder) if not cache \
        else "%s/%s/%s" % (vars.cache_folder, "results", folder)

    os.makedirs(folder, exist_ok=True)

    filename = "%s/%s" % (folder, task.model.model_name)
    res = {
        idx: {
            "result": [results],
            "task": task.to_dict()
        }
    }

    res[idx]['task']['random_seed'] = [task.random_seed]

    roc_res = {
        idx: {task.random_seed: roc_list}
    }

    try:
        results = dill.load(open(filename, "rb"))
        roc_results = dill.load(open(filename+".roc", "rb"))
        print("This id is already in results:")
        print(idx in results)
        if idx in results:
            print("random_seeds:")
            print(results[idx]['task']['random_seed'])
            if results[idx]['task'].get("random_seed") and \
                    task.random_seed not in results[idx]['task']['random_seed']:
                results[idx]['result'].extend(res[idx]['result'])
                results[idx]['task']['random_seed'] += [task.random_seed]
                roc_results[idx][task.random_seed] = roc_list
        else:
            results.update(res)
            roc_results.update(roc_res)

        if idx not in roc_results:
            roc_results.update(res)
        if roc_results.get(idx) and task.random_seed not in roc_results.get(idx):
            roc_results[idx][task.random_seed] = roc_list

        dill.dump(results, open(filename, "wb"))
        json.dump(results, open(filename + ".json", "w"))
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
    print(idx)
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
    set_random_seed(task.random_seed)

    task.model_config["device"] = task.device
    task.data_config["test"] = task.test

    data_card = DataConfig(**task.data_config)

    print("Loading data ...")
    data = load_cache(task.data_config)

    if not data:
        data = build_dataset.main(data_card)
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

    word_max_length = 512 if word_max_length > 512 else word_max_length

    if model_path:
        word_max_length = 512

    task.model_config["word_max_length"] = word_max_length
    model_card = task().model

    if not model:
        model = build_model.main(model_card)
        cache(model_card.to_dict(), model)

    if model_path:
        try:
            d = torch.load(model_path)
            datasets_trained = d['datasets_trained']
            state_dict = model.state_dict()
            state_dict.update(d['state_dict'])
            model.load_state_dict(state_dict)
            print("Model is updated from model_path.")
            print("Trained datasets: %s" % str(datasets_trained))
        except FileNotFoundError:
            datasets_trained = []
            print("Model isn't found at %s. Initiating one from config given." % model_path)


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

    epoch_remain = task.early_stop_epoch
    for i in range(task.epoch):
        print("Task running with: \n\t\t dataset %s" % task.data_config["huggingface_dataset_name"])
        print("\t\t model %s" % task.model_config['model_name'])
        pprint(model_card.to_dict())
        print(task.loss_func)

        torch.cuda.empty_cache()

        probs_test, preds_test, labels_test = None, None, None

        print("\t epoch %s" % str(i))
        loss_this_epoch_list = []
        model.train()
        clock_start = time()
        for batch in tqdm(train_dl, desc="Iteration"):
            texts, labels = batch
            labels = torch.tensor(labels)
            label_feature = train_tds.label_feature
            loss = model.batch_train(texts, labels, label_feature.names, task.loss_func, train_tds.multi_label)
            loss_this_epoch_list.append(loss.tolist())
            loss.backward()
            #
            # before_train = copy.deepcopy(dict(model.named_parameters()))
            optimizer.step()
            optimizer.zero_grad()
            # after_train = copy.deepcopy(dict(model.named_parameters()))

            # print("Printing Params at batch %i ..." % j)
            # input()
            # for name, param in before_train.items():
            #     if param.requires_grad and len(param.size()) > 1:
            #         print("\t"+name+" of size: %s" % str(param.size()))
            #         print("\tBefore train:")
            #         print(param)
            #         print("\n\tloss: %s" % str(loss))
            #         print("\n\tAfter train:")
            #         print(after_train[name])
            #         print("\tPress any button to continue ...")
            #         print("*********************************************************************")
            #         input()

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
        print(res["AUC"])
        print("Accuracy this epoch: %f." % (res["Accuracy"]))

        # If the accuracy is lower than previous results ...
        if acc_list and res["Accuracy"] <= acc_list[-1]:
            epoch_remain -= 1
            print("###################%i %i#######################" % (i, i + epoch_remain))
            if epoch_remain == 0:
                break
            continue

        epoch_remain = task.early_stop_epoch
        valid_i = i
        valid_res = res
        acc_list.append(res["Accuracy"])
        print(res['Classification report'])

    if not task.test:
        res = valid_res
        res['epochs'] = valid_i
        res['seconds_avg_epoch'] = clocks / (i + 1)
        res['accuracy_history_by_epoch'] = acc_list

        save_result(task, res, [labels_test.tolist(), probs_test.tolist()])

        print("\t Result saved ...")

        if model_path:
            state_dict = model.state_dict()
            if 'classifier.weight' in state_dict:
                del state_dict['classifier.weight']
                del state_dict['classifier.bias']
            if 'cls.weight' in state_dict:
                del state_dict['cls.weight']
                del state_dict['cls.bias']

            torch.save({
                "datasets_trained": datasets_trained + task.data.huggingface_dataset_name,
                "state_dict": state_dict,
                "model_config": task.model.to_dict(),
            }, model_path)

            print("\t Saving model at %s" % model_path)

        else:
            train_folder = "%s/%s" % (vars.trained_model_folder, task.model.model_name)

            os.makedirs(train_folder, exist_ok=True)
            file_name = task.idx()
            torch.save(model, "%s/%s" % (train_folder, file_name))

            print("\t Model saved ...")

    return model
