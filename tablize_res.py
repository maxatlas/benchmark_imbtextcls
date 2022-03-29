import copy
import os
import dill
import vars
import pandas as pd
import json
import pathlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from task_utils import get_res_df, get_auc_multiclass
from collections import defaultdict
from sklearn.metrics import roc_curve, auc


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

                matrix[i,j] = rowcol_sum/total_sum
                matrix[j,i] = colrow_sum/total_sum
    df = pd.DataFrame(matrix, index=levels, columns=levels)
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


def get_res_multi_file(folder=vars.results_folder):
    folder = Path(folder)
    datasets = [folder] if str(folder).split("/")[-1] in vars.dataset_names else [folder/ds for ds in os.listdir(folder)]
    by_ds, df = [], None
    for ds in tqdm(datasets):
        by_model =[]
        for model in os.listdir(ds):
            if model in vars.model_names:
                df = get_res_per_file(str(ds/model))
                by_model.append(df)
        if by_model:
            by_model = pd.concat(by_model, axis=1, levels=0)
            # by_model = by_model.groupby(by=by_model.columns.names, axis=1).mean()
            by_ds.append(by_model)
    if by_ds:
        df = pd.concat(by_ds, axis=0, levels=0, sort=False)
    return df


def get_res_per_file(file):
    def label_text_length(inputs):
        res = inputs[0]
        ds_name = "_".join(res['task']['data_config']['huggingface_dataset_name'])
        length_dict = vars.text_lengths[ds_name][0]
        if length_dict[0.75] <= 10:
            return ["small"]
        elif length_dict[0.25] > 10 and length_dict[0.75] <= 50:
            return ["medium"]
        elif length_dict[0.75] > 50:
            return ["large"]

    out = []
    pairs = dill.load(open(file, "rb")).items()
    for idx, results in pairs:
        index = {"dataset": (lambda x: ["_".join(x[0]['task']['data_config']['huggingface_dataset_name'])], [])}
        index["cls type"] = (lambda x: ["binary" if "_".join(x[0]['task']['data_config']['huggingface_dataset_name']) in vars.binary_ds else "multiclass"], [])
        index["label size"] = (lambda x: ["single" if "_".join(x[0]['task']['data_config']['huggingface_dataset_name']) not in vars.multilabel_ds else "multilabel"], [])
        index["proofread"] = (lambda x: ["proofread" if "_".join(x[0]['task']['data_config']['huggingface_dataset_name']) in vars.proofread_ds else "no proofread"], [])
        index["text_length"] = (label_text_length, [])

        header = {"model": (lambda x: [x[0]['task']['model_config']['model_name']], [])}
        header["num_layer"] = (lambda x: [x[0]['task']['model_config']['num_layers']], [])
        header['pretrained'] = (lambda x: [True if x[0]['task']['model_config']['pretrained_model_name'] else False], [])
        header['balance_strategy'] = (lambda x: [x[0]['task']['data_config']['balance_strategy']], [])
        header['loss function'] = (lambda x: [x[0]['task']['loss_func']], [])
        metrics = defaultdict(dict)

        auc_list = None
        try:
            auc_list = dill.load(open(file + ".roc", "rb"))[idx]
        except FileNotFoundError:
            print("No such file as %s.roc" % (file))
        except KeyError:
            print("No id (%s) in file %s.roc" % (idx, file))
        if results and auc_list:
            random_seeds = results['task']['random_seed']
            for i, random_seed in enumerate(random_seeds):
                if random_seed in auc_list:
                    # print(random_seeds, auc_list.keys())
                    metrics["AUC"][random_seed] = (lambda x: x[0][i]["AUC"], np.nan, []) if not auc_list else (lambda x: get_auc_multiclass(x[1], x[2]), auc_list[random_seed])
                else:
                    metrics["AUC"][random_seed] =(lambda x: [np.nan, np.nan], [])
                metrics['F1'][random_seed] = (lambda x: [x[0][i]['Macro-F1'], x[0][i]['Micro-F1']], [])
                metrics['Epochs'][random_seed] = (lambda x: [x[0][i]['epochs'],np.nan], [])
                metrics['Seconds/e'][random_seed] = (lambda x: [x[0][i]['seconds_avg_epoch'], np.nan], [])
            header[""] = (lambda x: ["Macro", "Micro"], [])
            index["Metrics"] = (lambda x: list(metrics.keys()), [])
            index["random_seed"] = (lambda x: random_seeds, [])

            df = get_res_df(results, header=header, index=index, metrics=metrics)
            out.append(df)
    out = pd.concat(out, axis=1, levels=1)
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
                        else:
                            for random_seed in roc_list:
                                if random_seed not in roc_results[idx]:
                                    roc_results[idx][random_seed] = roc_list[idx][random_seed]
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


if __name__ == "__main__":
    print("ye")
    # df = get_metric_value(["Accuracy", "Micro-F1"], res, "",
    #                       "1dce165479eece352b2887dcce81427e396cb5b6d2793c6825769122866aee9e", "")
    df = get_res_per_file("results/poem_sentiment/bert")
    # merge_res_from_sources(["../results_wiener", "results"], "merged")
    # suffix = "_balance_strategy_None"
    # # df = get_res("poem_sentiment"+suffix, "lstm", False, "gpt2", 1)
    # df = get_df_2d(["gpt2", "gpt2", "gpt2"], [1, 3, 5], True)
    # df.to_csv("merged/all.csv")
    # # df = get_df_2d(["bert-base-uncased", "xlnet-base-cased", "gpt2"], [1, 3, 5], True, vars.imbalanced_ds)
    # # df.to_csv("csv/imbalanced_dataset_overview.csv")
    #
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(df)
