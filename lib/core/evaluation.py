import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.config import config, update_config
from lib.datasets import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='setting.yaml')
    parser.add_argument('--add_note', type=str, default='T2')
    parser.add_argument('--gpu_list', default=0 , type=str , nargs='+')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_epoch', default=0, type=int)
    parser.add_argument('--save_interval', default=1, type=int)
    parser.add_argument('--dpath', default="", type=str)
    parser.add_argument('--gpath', default="", type=str)

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
    net.load_state_dict(pretrained_dict)
    print('modelG load from {}'.format(model_name))

def load_model_discriminator(net, load_path):
    model_name = load_path
    pretrained_dict = torch.load(model_name)
    net.load_state_dict(pretrained_dict)
    print('modelD load from {}'.format(model_name))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(666)
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


    model = models.get_baseline_net(config)
    model_D = models.D_net()
    gpus = [int(item) for item in args.gpu_list]
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
        shuffle=False,
        drop_last=True,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    for epoch in range(1):
        function.validate(config, val_loader, model, model_D, criterion, epoch, writer_dict=None,log_dir = None)

if __name__ == '__main__':
    main()










