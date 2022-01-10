import torch
import run_task

from vars import *
from Config import DataConfig, ModelConfig, TaskConfig
from torch.nn import BCEWithLogitsLoss


dataset_i = 3
dd = datasets_meta[dataset_i]
dd['test'] = 3
n_labels = 6

model_name = "cnn"
pretrained_tokenizer_name = "bert-base-uncased"
tokenizer_name = None
pretrained_model_name = None
emb_path = "parameters/emb_layer_glove"

md = {
    "model_name": model_name,
    "tokenizer_name": tokenizer_name,
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
                device="cuda:0",
                )

res = run_task.main(tc)
print(res)
