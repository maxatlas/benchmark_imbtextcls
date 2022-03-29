import vars
from tablize_res import *

import scipy.stats as st
import numpy as np
import vars
from task_utils import get_res_df, get_auc_multiclass, merge_single_res
from pathlib import Path
import os


def get_res_multi_file(folder):
    folder = Path(folder)
    datasets = [folder] if str(folder).split("/")[-1] in vars.dataset_names else [folder/ds for ds in os.listdir(folder)]
    by_ds = []
    for ds in datasets:
        by_model =[]
        for model in os.listdir(ds):
            print(ds/model)
            if model in vars.model_names:
                df = get_res_per_file(str(ds/model))
                by_model.append(df)
        if by_model:
            by_model = pd.concat(by_model, axis=1, levels=0)
            # by_model = by_model.groupby(by=by_model.columns.names, axis=1).mean()
            by_ds.append(by_model)
    if by_ds:
        df = pd.concat(by_ds, axis=0, levels=0)
    return df


def get_res_per_file(file):
    out = []
    pairs = dill.load(open(file, "rb")).items()
    for idx, results in pairs:
        index = {"dataset": (lambda x: ["_".join(x[0]['task']['data_config']['huggingface_dataset_name'])], [])}
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
            header[""] = (lambda x: ["Macro", "Micro"], [])
            index["Metrics"] = (lambda x: list(metrics.keys()), [])
            index["random_seed"] = (lambda x: random_seeds, [])

            df = get_res_df(results, header=header, index=index, metrics=metrics)
            out.append(df)
    out = pd.concat(out, axis=1, levels=1)
    return out


def test_get_metric_value():
    results = dill.load(open("results/ade_corpus_v2_Ade_corpus_v2_classification/bert", "rb"))
    idx = "fe5bb3e984bb389a4b2fc71f6b053cf1fac8ed17b26b8dea96ac0048e41e16fa"
    res = results[idx]
    df = get_metric_value(["AUC", "Micro-F1"], res, "", idx, [], [])
    return df


def test_get_result_per_file():
    metrics = ["AUC", "Micro-F1"]
    conditions = [
        {"task": {"data_config": {"huggingface_dataset_name": "ade_corpus_v2_Ade_corpus_v2_classification"}}},
        {"task": {
            "loss_func": ["TverskyLoss()", "DiceLoss()", "FocalLoss()", "CrossEntropyLoss()", "BCEWithLogitsLoss()"]}},
        {"task": {"data_config": {"balance_strategy": ["oversample", "undersample"]}}},
        # {"task": {"model_config": {"model_name": vars.model_names}}}
    ]
    result_file = "merged/ade_corpus_v2_Ade_corpus_v2_classification/bert"
    df = get_res_per_file(conditions, metrics, result_file)
    return df


# , {"task": {"data_config": {"huggingface_dataset_name": ["poem_sentiment"]}}}
def test_get_res():
    metrics = ["AUC"]
    # metrics = ["epochs"]
    conditions = [
        # {"task": {"loss_func": ["TverskyLoss()", "DiceLoss()", "FocalLoss()", "CrossEntropyLoss()", "BCEWithLogitsLoss()"]}},
        {"task": {"data_config": {"huggingface_dataset_name": vars.dataset_names}}},
        # {"task": {"model_config": {"model_name": vars.customized_model_names}}},
        {"task": {"data_config": {"balance_strategy": ["oversample", "undersample"]}}}
                  ]
    # conditions = [
        # {"task":{"model_config":{"model_name": vars.model_names}}},
        # {"task":{"data_config":{"balance_strategy":["oversample", 2157821"undersample"]}}}]
    out = get_res_test(metrics, conditions, "merged")
    return out


def get_ci(l):
    return st.t.interval(alpha=0.95, df=len(l.dropna()) - 1,
                         loc=np.mean(l.dropna()), scale=st.sem(l.dropna()))


if __name__ == "__main__":
    df = get_res_multi_file("../results_wiener")
    # df = test_get_metric_value()
    # df = test_get_result_per_file()
    with pd.option_context('display.max_columns', None, 'display.max_rows', None):
        print(df)
    # out = df.agg(lambda l: get_ci(l.dropna()), axis=0)
    # with pd.option_context('display.max_columns', None, 'display.max_rows', None):
    #     print(out)