import copy
import os
import dill
import vars
import pandas as pd
import json
import pathlib
import pingouin as pg
import numpy as np
from tqdm import tqdm
from pathlib import Path
from task_utils import get_res_df, get_auc_multiclass
from collections import defaultdict
from sklearn.metrics import roc_curve, auc

from hashlib import sha256


def create_matrix_from_df(df, level: int, axis: int):
    """

    :param df:
    :param level:
    :param axis:  1 for column, 0 for index
    :return:
    """
    levels = df.index.unique(level) if axis == 0 else df.columns.unique(level)
    names = df.index.names if axis == 0 else df.columns.names
    matrix = np.ones((len(levels), len(levels)))

    for i, key0 in enumerate(levels):
        data0 = df.xs(key0, level=names[level], axis=axis)
        for j, key1 in enumerate(levels):
            data1 = df.xs(key1, level=names[level], axis=axis)
            if i == j:
                matrix[i, j] = np.nan
            else:
                rowcol_sum, colrow_sum, total_sum = 0, 0, 0
                for key in data0.columns:
                    if key not in data1.columns:
                        continue
                    rowcol = (data0.loc[:, key] > data1.loc[:, key]).sum()
                    colrow = (data0.loc[:, key] < data1.loc[:, key]).sum()

                    rowcol_sum += rowcol
                    colrow_sum += colrow
                    total_sum += rowcol + colrow

                matrix[i, j] = rowcol_sum/total_sum
                matrix[j, i] = colrow_sum/total_sum
    df = pd.DataFrame(matrix, index=levels, columns=levels)
    return df


def hypo_test(df, level: int, axis: int):

    levels = df.index.unique(level) if axis == 0 else df.columns.unique(level)
    names = df.index.names if axis == 0 else df.columns.names
    matrix = np.ones((len(levels), len(levels)))

    for i, key0 in enumerate(levels):
        data0 = df.xs(key0, level=names[level], axis=axis).to_numpy().reshape(-1)
        data0 = data0[~np.isnan(data0)]
        for j, key1 in enumerate(levels):
            data1 = df.xs(key1, level=names[level], axis=axis).to_numpy().reshape(-1)
            data1 = data1[~np.isnan(data1)]

            if i == j:
                matrix[i, j] = np.nan

            else:
                if not data0.tolist() or not data1.tolist():
                    matrix[i, j] = np.nan
                    matrix[j, i] = np.nan
                else:
                    rowcol = pg.ttest(data0, data1, alternative="greater", correction=False).round(2)['p-val']
                    colrow = pg.ttest(data1, data0, alternative="greater", correction=False).round(2)['p-val']
                    matrix[i, j] = rowcol
                    matrix[j, i] = colrow
    df = pd.DataFrame(matrix, index=levels, columns=levels)
    df = df.dropna(1, "all")
    df = df.dropna(0, "all")
    return df


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


def get_res_multi_file(folder=vars.results_folder, df_path=None, save_path="merged.df"):
    folder = Path(folder)
    datasets = [folder] \
        if str(folder).split("/")[-1] in vars.dataset_names else [folder/ds for ds in os.listdir(folder)]
    df = None if not df_path else dill.load(open(df_path, "rb"))
    for ds in tqdm(datasets):
        for model in os.listdir(ds):
            if model in vars.model_names:
                print(str(ds/model))
                df = get_res_per_file(str(ds/model), out=df)
    if save_path:
        dill.dump(df, open(save_path, "wb"))
    return df


def get_res_per_file(file, out=None):
    pairs = dill.load(open(file, "rb")).items()
    for idx, results in pairs:
        header, index = vars.header_meta, vars.index_meta
        metrics = defaultdict(dict)
        auc_list = None
        try:
            auc_list = dill.load(open(file + ".roc", "rb"))[idx]
        except FileNotFoundError:
            print("No such file as %s.roc" % file)
        except KeyError:
            print("No id (%s) in file %s.roc" % (idx, file))
        if results and auc_list:
            random_seeds = results['task']['random_seed']
            for i, random_seed in enumerate(random_seeds):
                if random_seed in auc_list:
                    metrics["AUC"][random_seed] = (lambda x: x[0][i]["AUC"], np.nan, []) \
                        if not auc_list else (lambda x: get_auc_multiclass(x[1], x[2]), auc_list[random_seed])
                else:
                    metrics["AUC"][random_seed] = (lambda x: [np.nan, np.nan], [])
                metrics['F1'][random_seed] = (lambda x: [x[0][i]['Macro-F1'], x[0][i]['Micro-F1']], [])
                metrics['Epochs'][random_seed] = (lambda x: [x[0][i]['epochs'], np.nan], [])
                metrics['Seconds/e'][random_seed] = (lambda x: [x[0][i]['seconds_avg_epoch'], np.nan], [])
            index["metrics"] = {"values": metrics.keys(), "func": (lambda x: x[0], [])}
            index['random_seed'] = {"values": [29, 129, 444], "func": (lambda x: x[0], [])}

            out = get_res_df(results, header=header, index=index, metrics=metrics, out=out)
    return out


def merge_res_from_sources(folders, destination):
    os.makedirs(destination, exist_ok=True)
    for data in vars.dataset_names:
        for model in vars.model_names:
            file = Path(data, model)
            jfile = Path(data, model+".json")
            rfile = Path(data, model+".roc")
            results = defaultdict(dict)
            roc_results = defaultdict(dict)
            for folder in folders:
                try:
                    res = dill.load(open(folder/file, "rb"))
                    roc = dill.load(open(folder/rfile, "rb"))

                    print(folder/file)
                    for idx, value in res.items():
                        task = copy.deepcopy(value['task'])
                        roc_seed_list = roc.get(idx)
                        del task['random_seed']
                        del task['batch_size']
                        if "emb_path" in task["model_config"]:
                            del task['model_config']['emb_path']

                        if "word_max_length" in task['model_config']:
                            del task['model_config']['word_max_length']

                        idx = sha256(str(task).encode('utf-8')).hexdigest()
                        if idx in results:
                            for i, random_seed in enumerate(value['task']['random_seed']):
                                if random_seed not in results[idx]['task']['random_seed']:
                                    results[idx]['result'].append(value['result'][i])
                                    results[idx]['task']['random_seed'].append(value['task']['random_seed'][i])
                                    if roc_seed_list:
                                        roc_results[idx][random_seed] = roc_seed_list[random_seed]
                        else:
                            results.update({idx: value})
                            if roc_seed_list:
                                roc_results.update({idx: roc_seed_list})
                except FileNotFoundError:
                    continue
                except EOFError:
                    continue
            print("Saving results for %s %s" % (data, model))
            if results:
                print("\t Saving results")
                os.makedirs(destination+"/%s" % data, exist_ok=True)
                dill.dump(results, open(destination/file, "wb"))
                json.dump(results, open(destination/jfile, "w"))
            if roc_results:
                print("\t Saving ROC")
                dill.dump(roc_results, open(destination/rfile, "wb"))


if __name__ == "__main__":
    print("ye")
    # df = get_metric_value(["Accuracy", "Micro-F1"], res, "",
    #                       "1dce165479eece352b2887dcce81427e396cb5b6d2793c6825769122866aee9e", "")
    # df = get_res_per_file("merged/banking77/cnn")
    # merge_res_from_sources(["../results_wiener", "results", "../results_dracula"], "merged")
    from random_task import reformat_results
    # reformat_results("merged")
    # suffix = "_balance_strategy_None"
    # # df = get_res("poem_sentiment"+suffix, "lstm", False, "gpt2", 1)
    # df = get_df_2d(["gpt2", "gpt2", "gpt2"], [1, 3, 5], True)
    # df.to_csv("merged/all.csv")
    # # df = get_df_2d(["bert-base-uncased", "xlnet-base-cased", "gpt2"], [1, 3, 5], True, vars.imbalanced_ds)
    # # df.to_csv("csv/imbalanced_dataset_overview.csv")
    #
    # with pd.option_context('display.max_rows', None,
    #                        'display.max_columns', None,
    #                        'display.precision', 3,
    #                        ):
    #     print(df)
