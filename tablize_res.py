import copy
import os
import dill
import vars
import pandas as pd
import json
import pathlib
import numpy as np
import scipy.stats as st
from typing import List
from pathlib import Path
from task_utils import get_res_df, get_auc_multiclass
from collections import defaultdict
from sklearn.metrics import roc_curve, auc


def get_roc_curve_by_class(label_list: list, prob_list: list):
    roc_curve_by_class = {}
    num_label = len(label_list[0])

    prob_by_cls = {}
    for i in range(num_label):
        # label_list, prob_list
        prob_by_cls[i] = [[], []]

    for prob, label in zip(prob_list, label_list):
        for i in range(num_label):
            prob_by_cls[i][0].append(label[i] == 1)
            prob_by_cls[i][1].append(prob[i])

    for key, l in prob_by_cls.items():
        label_list, prob_list = l
        fpr, tpr, thresholds = roc_curve(label_list, prob_list, pos_label=1)
        fpr, tpr, thresholds = fpr.tolist(), tpr.tolist(), thresholds.tolist()
        roc_auc = auc(fpr, tpr)
        roc_curve_by_class[key] = {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}

    return roc_curve_by_class


def check_condition(condition: dict, res: dict):
    """

    :param condition: {"task" : {"data_config": {"balance_strategy": ["oversample", "undersample"]}}}
    :param res: {"task:{...}, "result":{...}}
    :return: True or False, {key: actual_value}
    """
    key, value = list(condition.keys())[0], list(condition.values())[0]
    res_value = res[key]
    if not condition:
        return True, {}
    if type(value) == dict:
        return check_condition(value, res_value)
    else:
        if key == "huggingface_dataset_name":
            res_value = "_".join(res_value)
        if type(value) == list:
            if type(res_value) == list:
                return res_value == value, {key: res_value}
            return res_value in value, {key: res_value}
        return res_value == value, {key: res_value}


def get_metric_value(metrics: list, res: dict, roc_file_path: str, idx: str, header, names):
    """
    should always return mean value and confidence intervals.

    :param model_id: "layer-n loss_function"
    :param metrics: ["precision f1"] Doesn't accept roc_curve though. Only result of a single value.
    :param res:
    :param roc_file_path:
    :param idx:
    :return: dataframe with metrics as index
    """
    results = res['result']
    data, out = defaultdict(list), []
    actual_metrics = copy.deepcopy(metrics)
    for result in results:
        for metric in metrics:
            if metric.lower() in ["roc curve"]:
                continue
            value = result.get(metric)

            if metric.lower() == "auc" and value == -1:
                if metric in actual_metrics:
                    actual_metrics.remove(metric)
                    actual_metrics.extend(["AUC Macro", "AUC Micro"])
                try:
                    roc = dill.load(open(roc_file_path, "rb")).get(idx)
                    if roc and type(roc) is list:
                        macro_auc, micro_auc = get_auc_multiclass(*roc)
                        data["AUC Macro"].append(macro_auc)
                        data["AUC Micro"].append(micro_auc)
                    else:
                        data["AUC Macro"].append(np.nan)
                        data["AUC Micro"].append(np.nan)
                except EOFError:
                    data["AUC Macro"].append(np.nan)
                    data["AUC Micro"].append(np.nan)
            else:
                data[metric].append(value)

    header = [[h] for h in header]
    header = pd.MultiIndex.from_product(header+[actual_metrics], names=names+["Metrics"])
    # for metric in metrics:
    #     l = data[metric]
    #     mean = np.mean(l)
    #     ci = st.t.interval(alpha=0.95, df=len(l) - 1, loc=mean, scale=st.sem(l))
    #     out.append([mean, ci])
    # print(out)

    # zip metric values
    out = list(zip(*list(data.values())))
    out = pd.DataFrame(out, columns=header)

    return out


def get_res_per_file(conditions: list, metrics: list, result_file: str):
    """
    Select metrics from results_old that satisfy conditions
    panda frame, organized by conditions
    :param res: a single result entity which corresponds to a task id: {task_id: {...}}
    :param conditions:
    :param metrics:
    :return:
    """
    out = defaultdict(list)

    results = dill.load(open(result_file, "rb"))
    for idx, res in results.items():
        if all([check_condition(condition, res)[0] for condition in conditions]):
            end_conditions = [check_condition(condition, res)[1] for condition in conditions]
            index = [list(condition.values())[0] for condition in end_conditions]
            names = [list(condition.keys())[0] for condition in end_conditions]
            value = get_metric_value(metrics, res, str(result_file) + ".roc", idx, index, names)
            out[str(value.columns.tolist())].append(value)

    rows = []
    for row in out:
        rows.append(pd.concat(out[row], ignore_index=True))

    out = None
    for row in rows:
        if out is None:
            out = row
        else:
            out = out.join(row)
    return out


def get_res_test(metrics: list, conditions: list = None, folder: str = vars.results_folder):
    out = []
    folder = pathlib.Path(folder)
    for dataset in os.listdir(folder):
        files = os.listdir(folder/dataset)
        for model in vars.model_names:
            if model not in files:
                continue
            out.append(get_res_per_file(conditions, metrics, folder/dataset/model))

    out = pd.concat(out, ignore_index=True, axis=0)
    return out


def merge_res_from_sources(folders, destination):

    os.makedirs(destination, exist_ok=True)
    for data in vars.dataset_names:
        for model in vars.model_names:
            file = Path(data, model)
            jfile = Path(data, model+".json")
            rfile = Path(data, model+".roc")
            results = {}
            roc_results = {}
            for folder in folders:
                try:
                    res = dill.load(open(folder/file, "rb"))
                    for idx, value in res.items():
                        if idx in results:
                            for i, random_seed in enumerate(value['task']['random_seed']):
                                if random_seed not in results['idx']['task']['random_seed']:
                                    results['idx']['result'].append(value['result'][i])
                                    results['idx']['task']['random_seed'].append(value['task']['random_seed'])

                        else:
                            results.update({idx: value})

                    roc = dill.load(open(folder/rfile, "rb"))
                    for idx, roc_list in roc.items():
                        if idx not in roc_results:
                            roc_results.update({idx: roc_list})
                except FileNotFoundError:
                    continue
                except EOFError:
                    continue
            if results:
                os.makedirs(destination+"/%s" % data, exist_ok=True)
                dill.dump(results, open(destination/file, "wb"))
                json.dump(results, open(destination/jfile, "w"))
            if roc_results:
                dill.dump(roc_results, open(destination/rfile, "wb"))


def get_res(data: str, model: str,
            pretrained_model: bool,
            pretrained_tokenizer: str = "",
            size: int = None,
            remove_linear: bool = True,
            loss_funcs=None):
    """
    1. Determine what results_old to select.
    2. Present.

    :param loss_funcs:
    :param data:
    :param model:
    :param pretrained_model:
    :param pretrained_tokenizer:
    :param size:
    :param remove_linear:
    :return:
    """
    if loss_funcs is None:
        loss_funcs = set()
    file_name = Path("merged", data, model)
    try:
        res_objects = dill.load(open(file_name, "rb"))
    except FileNotFoundError:
        print(file_name)
        return []

    out = []

    for idx, res in res_objects.items():
        if res["task"]["data_config"]["label_field"] == "product_category":
            continue
        model = res["task"]["model_config"]
        if pretrained_model:
            if model["pretrained_model_name"]:
                out.append(res)
        else:
            if model["pretrained_tokenizer_name"] == pretrained_tokenizer and \
                    model["num_layers"] == size and (model.get("disable_selfoutput") == remove_linear
                                                     or model.get("disable_selfoutput") is None):
                out.append(res)
        if not loss_funcs or res['task']['loss_func'] in loss_funcs:
            out.append(res)
    # out = merge_multi_res(out)
    return out


def get_df_2d(pretrained_tokenizers: List[str],
              sizes: List[int],
              pretrained_model: bool = False,
              datasets: list = None,
              models: list = None,
              loss_funcs=None):
    """

    :param pretrained_tokenizers:
    :param sizes:
    :param pretrained_model:
    :param datasets:
    :param models:
    :param loss_funcs: available options: ["CrossEntropyLoss()", "DiceLoss()", "TverskyLoss()", "FocalLoss()"]
    :return:
    """
    if loss_funcs is None:
        loss_funcs = []
    if not datasets:
        datasets = os.listdir(vars.results_folder)
    if not models:
        models = vars.model_names
    by_data = []
    size_pretrain = list(zip(sizes, pretrained_tokenizers))
    for data in datasets:
        by_model = []
        for model in models:
            res = []
            for size, pretrained_tokenizer in size_pretrain:
                res += get_res(data, model, False, pretrained_tokenizer, size, True,
                               loss_funcs=loss_funcs)
            if pretrained_model:
                res += get_res(data, model, True)
            df = [get_res_df(res0) for res0 in res]
            by_model.extend(df)

        if by_model:
            by_model = pd.concat(by_model, axis=1, levels=0)
            by_model = by_model.groupby(by=by_model.columns.names, axis=1).mean()
            by_data.append(by_model)

    if by_data:
        by_data = pd.concat(by_data, axis=0, levels=0)

    return by_data


if __name__ == "__main__":
    print("ye")
    # df = get_metric_value(["Accuracy", "Micro-F1"], res, "",
    #                       "1dce165479eece352b2887dcce81427e396cb5b6d2793c6825769122866aee9e", "")
    # merge_res_from_sources(["results_weiner", "results_old"], "merged")
    # suffix = "_balance_strategy_None"
    # # df = get_res("poem_sentiment"+suffix, "lstm", False, "gpt2", 1)
    # df = get_df_2d(["gpt2", "gpt2", "gpt2"], [1, 3, 5], True)
    # df.to_csv("merged/all.csv")
    # # df = get_df_2d(["bert-base-uncased", "xlnet-base-cased", "gpt2"], [1, 3, 5], True, vars.imbalanced_ds)
    # # df.to_csv("csv/imbalanced_dataset_overview.csv")
    #
    # # with pd.option_context('display.max_rows', None,
    # #                        'display.max_columns', None,
    # #                        'display.precision', 3,
    # #                        ):
    # #     print(df)
