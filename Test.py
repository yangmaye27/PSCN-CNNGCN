import os
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.config import config, update_config
from lib.datasets import get_dataset

dataset_type = get_dataset(config)
train_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * 2,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        drop_last=True,
        pin_memory=config.PIN_MEMORY)
for i, (inp_RN, inp_Quant, inp_Vis, inp_guid, inp_Nd, target_256, target_64_jc_all, target_64_st, Slopmap) in enumerate(
        train_loader):
    print(i)