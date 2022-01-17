import torch
import run_task
import argparse

from vars import *
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from Config import TaskConfig

import build_dataset
from Config import DataConfig
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_i',
                        '-i',
                        default=None,
                        required=True,
                        type=int,
                        help="nth dataset from vars.datasets_meta")
parser.add_argument("--device",
                    default="cuda:0",
                    type=str,
                    help="which device to run on: cpu/cuda:n")
parser.add_argument('--model',
                    '-m',
                    type=str,
                    required=True,)
parser.add_argument('--model_pretrained',
                    '-p',
                    type=str,
                    default="")
parser.add_argument('--tokenizer',
                    '-z',
                    type=str,
                    default="")
parser.add_argument('--tokenizer_pretrained',
                    '-x',
                    type=str,
                    default="")
parser.add_argument('--embedder',
                    '-b',
                    type=str,
                    default="glove")
parser.add_argument("--test",
                    '-t',
                    default=0,
                    type=int)
parser.add_argument('--epoch',
                    '-e',
                    type=int,
                    default=3)

args = parser.parse_args()

dataset_i = args.dataset_i
dd = datasets_meta[dataset_i]
# dc = DataConfig(**dd)
# train_tds, _, _, split_info = build_dataset.main(dc)
# dl = DataLoader(train_tds, batch_size=100, shuffle=True)
# next(iter(dl))

model_name = args.model
pretrained_tokenizer_name = args.tokenizer_pretrained
tokenizer_name = args.tokenizer
pretrained_model_name = args.model_pretrained
emb_path = "%s/emb_layer_%s" % (parameter_folder, args.embedder)
loss = BCEWithLogitsLoss if dd.get("multi_label") else CrossEntropyLoss

md = {
    "model_name": model_name,
    "tokenizer_name": tokenizer_name,
    "pretrained_model_name": pretrained_model_name,
    "pretrained_tokenizer_name": pretrained_tokenizer_name,
    "emb_path": emb_path
}

optimizer = torch.optim.AdamW
tc = {
    "model_config": md,
    "data_config": dd,
    "batch_size": 100,
    "loss_func": loss(),
    "optimizer": optimizer,
    "device": args.device,
    "test": None if not args.test else args.test,
    "epoch": args.epoch,
}

task = TaskConfig(**tc)
res = run_task.main(task)
print(res)
