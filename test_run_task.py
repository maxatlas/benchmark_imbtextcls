import torch
import run_task
import vars
import os
from vars import *
from torch.nn import BCEWithLogitsLoss

# os.environ['TRANSFORMERS_CACHE'] = vars.hf_cache_folder+"/modules"
# os.environ['HF_DATASETS_CACHE'] = vars.hf_cache_folder+'/datasets'


dataset_i = 1
dd = datasets_meta[dataset_i]

model_name = "roberta"
pretrained_tokenizer_name = "roberta-base"
tokenizer_name = None
pretrained_model_name = ""
emb_path = "params/emb_layer_glove"

md = {
    "model_name": model_name,
    "tokenizer_name": tokenizer_name,
    "pretrained_model_name": pretrained_model_name,
    "pretrained_tokenizer_name": pretrained_tokenizer_name,
    "emb_path": emb_path
}

optimizer = torch.optim.AdamW
tc = {
    "model_config_dict": md,
    "data_config_dict": dd,
    "batch_size": 100,
    "loss_func": BCEWithLogitsLoss(),
    "optimizer": optimizer,
    "device": "cuda:0",
    "test": 3,
}


res = run_task.main(tc)
print(res)
