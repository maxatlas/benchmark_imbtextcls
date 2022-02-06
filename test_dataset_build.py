import build_dataset
from Config import DataConfig
from torch.utils.data import DataLoader
from vars import *

dataset_i = 19
dd = datasets_meta[dataset_i]
dc = DataConfig(**dd)
train_tds, _, _, split_info = build_dataset.main(dc)
dl = DataLoader(train_tds, batch_size=100, shuffle=True)
next(iter(dl))