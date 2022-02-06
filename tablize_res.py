import os
import dill
import vars
import pandas as pd

from typing import List
from pathlib import Path
from task_utils import get_res_df, merge_multi_res


def get_res(data: str, model: str,
            pretrained_model: bool,
            pretrained_tokenizer: str = "",
            size: int = None,):
    file_name = Path("merged", data, model)
    res_objects = dill.load(open(file_name, "rb"))

    pretrains = []
    non_pretrains = []

    for idx, res in res_objects.items():
        model = res["task"]["model_config"]
        if model["pretrained_model_name"]:
            if pretrained_model:
                pretrains.append(res)
        else:
            if model["pretrained_tokenizer_name"] == pretrained_tokenizer and \
                    model["num_layers"] == size:
                non_pretrains.append(res)
    out = merge_multi_res(pretrains) + merge_multi_res(non_pretrains)
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
            for size, pretrained_tokenizer in size_pretrain:
                res = get_res(data, model, pretrained_model, pretrained_tokenizer, size)
                df = [get_res_df(res0) for res0 in res]
                by_model.extend(df)
        if by_model:
            by_model = pd.concat(by_model, axis=1, levels=0)
            by_data.append(by_model)

    if by_data:
        by_data = pd.concat(by_data, axis=0, levels=0)

    return by_data


if __name__ == "__main__":
    suffix = "_balance_strategy_None"
    # df = get_res("poem_sentiment"+suffix, "bert", True, "bert-base-uncased", 1)
    df = get_df_2d(["bert-base-uncased", "xlnet-base-cased", "gpt2"], [1, 3, 5], True, ["poem_sentiment"+suffix])
    #
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(df)
