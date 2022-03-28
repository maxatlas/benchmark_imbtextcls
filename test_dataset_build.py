import build_dataset
from Config import DataConfig
from torch.utils.data import DataLoader
from vars import *
from task_utils import set_random_seed

random_seed = 129
set_random_seed(random_seed)
dataset_i = 13
dd = datasets_meta[dataset_i]
dd['balance_strategy'] = None
dc = DataConfig(**dd)

train_tds, test_tds, val_tds, split_info = build_dataset.main(dc)
import dill
dill.dump({"train":train_tds, "test":test_tds, "val":val_tds}, open(".cache/ade_%s_%i"% (dd['balance_strategy'], random_seed), "wb"))
dill.load(open(".cache/ade_%s_%i"% (dd['balance_strategy'], random_seed), "rb"))
dl = DataLoader(train_tds, batch_size=20, shuffle=True)
next(iter(dl))
