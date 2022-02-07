import os
import dill
import vars
import pandas as pd

from typing import List
from pathlib import Path
from task_utils import get_res_df, merge_multi_res
from random_task import merge_res_from_sources

def get_res(data: str, model: str,
            pretrained_model: bool,
            pretrained_tokenizer: str = "",
            size: int = None,):
    file_name = Path("merged", data, model)
    try:
        res_objects = dill.load(open(file_name, "rb"))
    except FileNotFoundError:
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
                    model["num_layers"] == size:
                out.append(res)
    # out = merge_multi_res(out)
    return out


def get_df_2d(pretrained_tokenizers: List[str],
              sizes: List[int],
              pretrained_model: bool = False,
              datasets: list = None,
              models: list = None):
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
                res += get_res(data, model, False, pretrained_tokenizer, size)
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
    # df = get_res("poem_sentiment"+suffix, "bert", True, "bert-base-uncased", 1)
    df = get_df_2d(["bert-base-uncased", "xlnet-base-cased", "gpt2"], [1, 3, 5], True)
    df.to_csv("csv/all.csv")
    # df = get_df_2d(["bert-base-uncased", "xlnet-base-cased", "gpt2"], [1, 3, 5], True, vars.imbalanced_ds)
    # df.to_csv("csv/imbalanced_dataset_overview.csv")

    # with pd.option_context('display.max_rows', None,
    #                        'display.max_columns', None,
    #                        'display.precision', 3,
    #                        ):
    #     print(df)
