import torch
import run_task
from vars import *
from torch.nn import BCEWithLogitsLoss
from Config import TaskConfig
# os.environ['TRANSFORMERS_CACHE'] = vars.hf_cache_folder+"/modules"
# os.environ['HF_DATASETS_CACHE'] = vars.hf_cache_folder+'/datasets'


dataset_i = 1
dd = datasets_meta[dataset_i]

model_name = "cnn"
pretrained_tokenizer_name = "bert-base-cased"
tokenizer_name = None
pretrained_model_name = ""
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
    "device": "cpu",
    "test": 3,
}

task = TaskConfig(**tc)
res = run_task.main(task)
print(res)
