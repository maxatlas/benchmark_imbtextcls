import torch
import run_task
from vars import *
from torch.nn import BCEWithLogitsLoss
from Config import TaskConfig

import build_dataset
from Config import DataConfig
from torch.utils.data import DataLoader
# os.environ['TRANSFORMERS_CACHE'] = vars.hf_cache_folder+"/modules"
# os.environ['HF_DATASETS_CACHE'] = vars.hf_cache_folder+'/datasets'


dataset_i = 1
dd = datasets_meta[dataset_i]
# dc = DataConfig(**dd)
# train_tds, _, _, split_info = build_dataset.main(dc)
# dl = DataLoader(train_tds, batch_size=100, shuffle=True)
# next(iter(dl))

model_name = "xlnet"
pretrained_tokenizer_name = ""
tokenizer_name = ""
pretrained_model_name = "xlnet-base-cased"
emb_path = "%s/emb_layer_glove" % parameter_folder

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
    "loss_func": BCEWithLogitsLoss(),
    "optimizer": optimizer,
    "device": "cuda:0",
    "test": None,
    "epoch": 1,
}

task = TaskConfig(**tc)
res = run_task.main(task)
print(res)
