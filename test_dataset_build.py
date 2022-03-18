import build_dataset
from Config import DataConfig
from torch.utils.data import DataLoader
from vars import *

dataset_i = 11
dd = datasets_meta[dataset_i]
dd['balance_strategy'] = "oversample"

dc = DataConfig(**dd)
train_tds, test_tds, val_tds, split_info = build_dataset.main(dc)
dl = DataLoader(train_tds, batch_size=20, shuffle=True)
next(iter(dl))
