from torch.utils.data import DataLoader
from lib.config import config, update_config
from lib.datasets import get_dataset

dataset_type = get_dataset(config)
train_loader = DataLoader(
    dataset=dataset_type(config,
                         is_train=True),
    batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len([0]),
    shuffle=config.TRAIN.SHUFFLE,
    num_workers=config.WORKERS,
    drop_last=True,
    pin_memory=config.PIN_MEMORY)

for i, (
        inp_RN, inp_Quant, inp_Vis, inp_guid_ss, inp_Nd, target_256, target_64_jc_all, target_64_st,
        Slopmap, nodes, adjacency, edge_attr) in enumerate(
    train_loader):
    print(123)