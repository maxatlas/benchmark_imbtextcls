import torch
import vars
import argparse

from torch.nn import BCEWithLogitsLoss

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_i',
                    default=None,
                    required=True,
                    type=int,
                    help="nth dataset from vars.datasets_meta")
parser.add_argument("--device",
                    default="cuda:0",
                    type=str,
                    help="which device to run on: cpu/cuda:n")

args = parser.parse_args()
dc = vars.datasets_meta[args.dataset_i]

print("Pretrained models ...")

task_cards = []

for model, pretrain in list(zip(vars.transformer_names,
                                vars.transformer_pretrain)):
    mc = {
        "model_name": model,
        "pretrained_model_name": pretrain,
    }
    tc = {
        "data_config_dict": dc,
        "model_config_dict": mc,
        "batch_size": 100,
        "loss_func": BCEWithLogitsLoss(),
        "device": args.device,
        "optimizer": torch.optim.AdamW,
    }

    task_cards.append(tc)
