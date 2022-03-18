import os
import dill
import vars
import pandas as pd
import json

from typing import List
from pathlib import Path
from task_utils import get_res_df, merge_multi_res


def merge_res_from_sources(folders, destination):

    os.makedirs(destination, exist_ok=True)
    for data in vars.dataset_names:
        data = data + "_balance_strategy_None"
        for model in vars.model_names:
            file = Path(data, model)
            jfile = Path(data, model+".json")
            results = {}
            for folder in folders:
                try:
                    res = dill.load(open(folder/file, "rb"))
                    for idx, value in res.items():
                        if idx in results:
                            results[idx]["result"].extend(value["result"])
                        else:
                            results.update({idx: value})
                except FileNotFoundError:
                    continue
                except EOFError:
                    continue
            if results:
                os.makedirs(destination+"/%s" % data, exist_ok=True)
                dill.dump(results, open(destination/file, "wb"))
                json.dump(results, open(destination/jfile, "w"))


def get_res(data: str, model: str,
            pretrained_model: bool,
            pretrained_tokenizer: str = "",
            size: int = None,
            remove_linear: bool = True,
            loss_funcs=None):
    """
    1. Determine what results to select.
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
    # merge_res_from_sources(["res_uq", "results"], "merged")
    suffix = "_balance_strategy_None"
    # df = get_res("poem_sentiment"+suffix, "lstm", False, "gpt2", 1)
    df = get_df_2d(["gpt2", "gpt2", "gpt2"], [1, 3, 5], True)
    df.to_csv("merged/all.csv")
    # df = get_df_2d(["bert-base-uncased", "xlnet-base-cased", "gpt2"], [1, 3, 5], True, vars.imbalanced_ds)
    # df.to_csv("csv/imbalanced_dataset_overview.csv")

    # with pd.option_context('display.max_rows', None,
    #                        'display.max_columns', None,
    #                        'display.precision', 3,
    #                        ):
    #     print(df)
