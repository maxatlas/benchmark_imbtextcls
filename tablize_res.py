import os
import dill
import vars
import pandas as pd

from pathlib import Path
from task_utils import get_res_df


def get_res(data: str, model: str,
            pretrained_model: str,
            pretrained_tokenizer: str,
            size: int,):
    file_name = Path(vars.results_folder, data, model)
    res_objects = dill.load(open(file_name, "rb"))

    for idx, res in res_objects.items():
        model = res["task"]["model_config"]
        if model["pretrained_model_name"] == pretrained_model and \
                model["pretrained_tokenizer_name"] == pretrained_tokenizer and \
                model["num_layers"] == size:

            return res


def get_res_2d(pretrained_model: str,
               pretrained_tokenizer: str,
               size: int,
               datasets: list = [],
               models: list = []):
    if not datasets:
        datasets = os.listdir(vars.results_folder)
    if not models:
        models = vars.model_names

    by_data = []
    for data in datasets:
        by_model = []
        for model in models:
            # print(data, model)
            by_model.append(get_res_df(get_res(data, model, pretrained_model, pretrained_tokenizer, size)))

        by_model = pd.concat(by_model, axis=1, levels=0)
        by_data.append(by_model)
    by_data = pd.concat(by_data, axis=0, levels=0)

    return by_data


if __name__ == "__main__":
    suffix = "_balance_strategy_None"
    df = get_res_2d("", "bert-base-uncased", 1, ["sms_spam_balance_strategy_None", "sst_balance_strategy_None", "ade_corpus_v2_Ade_corpus_v2_classification_balance_strategy_None"],
                    ["bert", ])
                     # "gpt2","xlnet", "lstm", "lstmattn", "cnn", "rcnn", "han", "mlp"])
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(df)
