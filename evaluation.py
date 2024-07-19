import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
import random

from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.utils import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='setting.yaml')
    parser.add_argument('--add_note', type=str, default='T3_test')
    parser.add_argument('--gpu_list', default=0 , type=str , nargs='+')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_epoch', default=0, type=int)
    parser.add_argument('--save_interval', default=1, type=int)
    parser.add_argument('--dpath', default="./models/Model3DisGraph.pth", type=str)
    parser.add_argument('--gpath', default="./models/Model3GenGraph.pth", type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args

def adjust_learning_rate(lr_in, optimizer, epoch):
    """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
    if epoch < 30:
        lr = lr_in * (0.5 ** (epoch // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def load_model_generator(net, load_path):
    model_name = load_path
    pretrained_dict = torch.load(model_name)

    net.load_state_dict(pretrained_dict,strict=False)
    print('modelG load from {}'.format(model_name))

def load_model_discriminator(net, load_path):
    model_name = load_path
    pretrained_dict = torch.load(model_name)
    net.load_state_dict(pretrained_dict,strict=False)
    print('modelD load from {}'.format(model_name))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(666)


def collate_fn(batch):

    Sample_In_RN = torch.stack([item[0] for item in batch], dim=0)
    Sample_In_Quant_temp = torch.stack([item[1] for item in batch], dim=0)
    Sample_In_Vis_temp = torch.stack([item[2] for item in batch], dim=0)
    Sample_In_Guid = torch.stack([item[3] for item in batch], dim=0)
    Sample_In_ND = torch.stack([item[4] for item in batch], dim=0)
    Sample_GT256 = torch.stack([item[5] for item in batch], dim=0)
    Sample_GT64JC_ALL = torch.stack([item[6] for item in batch], dim=0)
    Sample_GT64ST = torch.stack([item[7] for item in batch], dim=0)
    Slopmap = torch.stack([item[8] for item in batch], dim=0)
    # graph_batch = Batch.from_data_list([item[9] for item in batch])
    # Graph data preparation
    node_features = [item[9].x for item in batch]
    edge_index = [item[9].edge_index for item in batch]
    edge_attr = [item[9].edge_attr for item in batch]

    max_num_nodes = max([nf.size(0) for nf in node_features])
    max_num_edges = max([ei.size(1) if ei.numel() > 1 else 0 for ei in edge_index])

    # Padding node features and edge attributes
    padded_node_features = []
    node_masks = []

    for nf in node_features:
        num_nodes = nf.size(0)
        padded_nf = torch.cat([nf, torch.zeros(max_num_nodes - num_nodes, 2)], dim=0)
        padded_node_features.append(padded_nf)
        node_mask = torch.cat([torch.ones(num_nodes), torch.zeros(max_num_nodes - num_nodes)], dim=0)
        node_masks.append(node_mask)

    padded_edge_index = []
    padded_edge_attr = []
    edge_masks = []

    for ei, ea in zip(edge_index, edge_attr):
        num_edges = ei.size(1) if ei.numel() > 0 else 0
        if num_edges > 0:
            padded_ei = torch.cat([ei, torch.zeros(2, max_num_edges - num_edges, dtype=torch.long)], dim=1)
            padded_ea = torch.cat([ea, torch.zeros(max_num_edges - num_edges, 1)], dim=0)
        else:
            padded_ei = torch.zeros(2, max_num_edges, dtype=torch.long)
            padded_ea = torch.zeros(max_num_edges, 1)
        padded_edge_index.append(padded_ei)
        padded_edge_attr.append(padded_ea)
        edge_mask = torch.cat([torch.ones(num_edges), torch.zeros(max_num_edges - num_edges)], dim=0)
        edge_masks.append(edge_mask)

    data_list = [Data(x=nf, edge_index=ei, edge_attr=ea,node_mask=nm,edge_mask=em) for nf, ei, ea,nm,em in
                 zip(padded_node_features, padded_edge_index, padded_edge_attr,node_masks,edge_masks)]
    # graph_batch = Batch.from_data_list(data_list)
    # graph_batch.node_masks = torch.stack(node_masks, dim=0)
    # graph_batch.edge_masks = torch.stack(edge_masks, dim=0)
    return Sample_In_RN, Sample_In_Quant_temp, Sample_In_Vis_temp, Sample_In_Guid, Sample_In_ND, \
        Sample_GT256, Sample_GT64JC_ALL, Sample_GT64ST, Slopmap, data_list

def main():
    args = parse_args()
    if args.add_note.find("T2") != -1:
        from lib.models_baseline import baseline_modelT2 as models
        from lib.core import function_baseline_T2 as function
    elif args.add_note.find("T1") != -1:
        from lib.models_baseline import baseline_modelT1 as models
        from lib.core import function_baseline_T1 as function
    elif args.add_note.find("T3") != -1:
        from lib.models_baseline import baseline_modelT3 as models
        from lib.core import function_baseline_T3 as function

    save_dir_name = os.path.basename(args.cfg).split('.')[0] + "_" +args.add_note
    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, save_dir_name, 'test')


    model = models.get_baseline_net(config)
    model_D = models.D_net()
    gpus = [0]
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    model_D = nn.DataParallel(model_D, device_ids=gpus).cuda()
    criterion = torch.nn.MSELoss(size_average=True).cuda()

    load_model_generator(model ,args.gpath)
    load_model_discriminator(model_D, args.dpath)
    dataset_type = get_dataset(config)
    val_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=True,
        drop_last=True,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_fn
    )
    for epoch in range(1):
        function.validate(config, val_loader, model, model_D, criterion, epoch, writer_dict=None,log_dir = final_output_dir)

if __name__ == '__main__':
    main()










