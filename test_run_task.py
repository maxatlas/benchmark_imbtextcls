import torch
import run_task

from vars import *
from Config import DataConfig, ModelConfig, TaskConfig
from torch.nn import BCEWithLogitsLoss


dataset_i = 3
dd = datasets_meta[dataset_i]
dd['test'] = 3
n_labels = 6

model_name = "roberta"
pretrained_tokenizer_name = None
tokenizer_name = None
pretrained_model_name = "roberta-base"
emb_path = "params/emb_layer_glove"

md = {
    "model_name": model_name,
    "tokenizer_name": tokenizer_name,
    "pretrained_model_name": pretrained_model_name,
    "pretrained_tokenizer_name": pretrained_tokenizer_name,
    "n_labels": n_labels,
    "word_max_length": 50,
    "emb_path": emb_path
}

optimizer = torch.optim.AdamW
tc = TaskConfig(dd, md,
                batch_size=100,
                test=3,
                loss_func=BCEWithLogitsLoss(),
                optimizer=optimizer,
                device="cpu",
                )

res = run_task.main(tc)
print(res)
