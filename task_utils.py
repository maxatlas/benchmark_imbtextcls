import torch
import random
import pandas as pd
import numpy as np

import scipy.stats as st
from sklearn.metrics import (
    f1_score,
    roc_curve,
    auc,
    recall_score,
    precision_score,
    classification_report,
    accuracy_score,
    roc_auc_score
)


def metrics_frame(probs, preds, labels, label_names):
    accuracy = accuracy_score(labels, preds)
    recall_micro = recall_score(labels, preds, average="micro")
    recall_macro = recall_score(labels, preds, average="macro")
    precision_micro = precision_score(labels, preds, average="micro")
    precision_macro = precision_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")

    cr = classification_report(labels, preds, target_names=label_names, output_dict=True)

    fpr, tpr, thresholds, auc_res = [-1] * 4

    if len(label_names) == 2:
        labels = [label.tolist().index(1) for label in labels]
        probs = torch.sigmoid(torch.tensor(probs))[:, 1]

        fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
        fpr, tpr, thresholds = fpr.tolist(), tpr.tolist(), thresholds.tolist()

        auc_res = auc(fpr, tpr)
    else:
        try:
            auc_res = get_auc_multiclass(labels, probs)
        except Exception:
            pass

    model_metrics = {
        "Accuracy": accuracy,
        "Precision, Micro": precision_micro,
        "Precision, Macro": precision_macro,
        "Micro-Recall": recall_micro,
        "Macro-Recall": recall_macro,
        "Micro-F1": f1_micro,
        "Macro-F1": f1_macro,
        "ROC curve": (fpr, tpr, thresholds),
        "AUC": auc_res,
        "Classification report": cr,
    }

    return model_metrics


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def merge_multi_res(results):
    if results:
        target_res = {"result":[]}
        for res in results:
            target_res["task"] = res["task"]
            target_res["result"].extend(res["result"])
        target_res['result'] = [merge_single_res(target_res)]
        return [target_res]

    return []


def merge_single_res(info):
    results = info['result']
    out = {}
    for key in results[0]:
        values = []
        for res in results:
            val = res[key]
            if type(val) is dict:
                val = pd.DataFrame.from_dict(val)
            values.append(val)
        if type(values[0]) is pd.core.frame.DataFrame:
            out[key] = sum(values)/len(values)
            out[key].fillna(0, inplace=True)
            out[key] = out[key].to_dict()
        else:
            if key == "ROC curve" and type(values[0]) == tuple:
                values = values[0]
            values = torch.tensor(values).view(-1)
            value = sum(values)/len(values)
            out[key] = float(value)
    return out


def get_res_df_fixed(info):
    if not info:
        return pd.DataFrame([])
    results = {}

    res = merge_single_res(info)
    task = info['task']
    model_name = task['model_config']['model_name']
    data_name = "_".join(task['data_config']['huggingface_dataset_name'])

    num_labels = task['model_config']['num_labels']
    cr = res["Classification report"]
    label_names = list(cr.keys())[:num_labels]

    for i, key in enumerate(label_names):
        results[key] = [cr[key]['f1-score'], int(cr[key]['support'])]

    results["macro"] = [res["Macro-F1"], np.nan]
    results["micro"] = [res["Micro-F1"], np.nan]
    results["weighted"] = [cr["weighted avg"]['f1-score'], np.nan]
    results["accuracy"] = [res['Accuracy'], np.nan]
    results["avg_sec_per_epoch"] = [res["seconds_avg_epoch"], np.nan]
    results["total_epochs"] = [res["epochs"]+1, np.nan]

    data = np.array(list(results.values()))

    pretrained = task['model_config']['pretrained_model_name']

    model_id = "%s%s" % ("pretrained" if pretrained else "",
                        "layer-" + str(task['model_config']['num_layers']) if not pretrained else "")
    # model_id = "%s%s" %(model_id, "" if 'disable_intermediate' in task['model_config'] and task['model_config']['disable_intermediate'] else "-no-linear")

    header = pd.MultiIndex.from_product([[model_name], ["%s" % model_id], ["f1-score", "support"]],
                                        names=["Model", "num_layer", "Metrics"])
    index = pd.MultiIndex.from_product([[data_name], list(results.keys())],
                                       names=["Dataset", "Categories"])
    df = pd.DataFrame(data, columns=header, index=index)
    return df


def get_res_df(info, index, header, metrics):
    """

    :param info: res = dill.load(filename), info = res[id]
    :param index: {names: (func, [input])} name, value
    :param header: {names: (func, [input])}
    :param metrics: {names: (func, [input])}
    :return:
    """
    if not info:
        return pd.DataFrame([])


    header = pd.MultiIndex.from_product(
        [func([info, *func_input]) for func, func_input in header.values()],
        names=list(header.keys()))

    index = pd.MultiIndex.from_product(
        [func([info, *func_input]) for func, func_input in index.values()],
        names=list(index.keys()))

    out = []
    res = info['result']
    for metric in metrics.values():
        for name, func in metric.items():
            row = func[0]([res, *func[1]])
            out.append(row)

    out = np.array(out)
    df = pd.DataFrame(out, columns=header, index=index.unique())
    # out.append(df)
    # out = pd.concat(out)

    return df


def get_auc_multiclass(label_list: list, prob_list: list, multiclass: bool = True):
    """

    :param label_list:[l.index(1) for l in label_list]
    :param prob_list:
    :param multiclass:
    :return: [auc macro, auc micro]
    """
    return [
        roc_auc_score(label_list, prob_list, multi_class="ovr" if multiclass else "ovo", average="macro"),
        roc_auc_score(label_list, prob_list, multi_class="ovr" if multiclass else "ovo", average="micro")]


def get_ci(l):
    return st.t.interval(alpha=0.95, df=len(l.dropna()) - 1,
                         loc=np.mean(l.dropna()), scale=st.sem(l.dropna()))


if __name__ == "__main__":
    import dill

    infos = list(dill.load(open("results_old/sms_spam_balance_strategy_np.nan/bert", "rb")).values())
    dfs = [get_res_df(info) for info in infos]
    pd.concat(dfs, axis=0, levels=0)  # different datasets
    out = pd.concat(dfs, axis=1, levels=0)  # different models

    print(out)
