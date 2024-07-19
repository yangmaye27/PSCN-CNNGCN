import os
import pprint
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.utils import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='setting.yaml')
    parser.add_argument('--add_note', type=str, default='T1')
    parser.add_argument('--gpu_list', default=0 , type=str , nargs='+')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_epoch', default=0, type=int)
    parser.add_argument('--save_interval', default=1, type=int)
    args = parser.parse_args()
    update_config(config, args)
    return args

def adjust_learning_rate(lr_in, optimizer, epoch):
    """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
    if epoch < 30:
        lr = lr_in * (0.5 ** (epoch // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def save_load_model_generator(net, epoch, config, final_output_dir, load_flag=False):
    model_name = 'modelG_epoch%d_batchsize%d.pth' % (epoch, config.TRAIN.BATCH_SIZE_PER_GPU)
    model_name = os.path.join(final_output_dir, model_name)
    if load_flag:
        pretrained_dict = torch.load(model_name)
        net.load_state_dict(pretrained_dict)
        print('model load from {}'.format(model_name))
    else:
        torch.save(net.state_dict(), model_name)
        print('The trained model is successfully saved at epoch %d' % (epoch))

def save_load_model_discriminator(net, epoch, config, final_output_dir, load_flag=False):
    """Save the model at "checkpoint_interval" and its multiple"""
    model_name = 'modelD_epoch%d_batchsize%d.pth' % (epoch, config.TRAIN.BATCH_SIZE_PER_GPU)
    model_name = os.path.join(final_output_dir, model_name)
    if load_flag:
        pretrained_dict = torch.load(model_name)
        net.load_state_dict(pretrained_dict)
        print('model load from {}'.format(model_name))
    else:
        torch.save(net.state_dict(), model_name)
        print('The trained model is successfully saved at epoch %d' % (epoch))
import random
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

    save_dir_name = os.path.basename(args.cfg).split('.')[0] + "_" +args.add_note
    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, save_dir_name, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = models.get_baseline_net(config)
    model_D = models.D_net()
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    gpus = [int(item) for item in args.gpu_list]
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    model_D = nn.DataParallel(model_D, device_ids=gpus).cuda()
    criterion = torch.nn.MSELoss(size_average=True).cuda()
    begin_epoch = config.TRAIN.BEGIN_EPOCH

    if args.resume:
        load_epoch = args.load_epoch
        begin_epoch = load_epoch
        save_load_model_generator(model, load_epoch, config, final_output_dir, load_flag=True)
        save_load_model_discriminator(model_D, load_epoch, config, final_output_dir, load_flag=True)
        print("load from:", final_output_dir, load_epoch)
    optimizer_g = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.5, 0.999), weight_decay=0)
    optimizer_d = torch.optim.Adam(model_D.parameters(), lr=2e-5, betas=(0.5, 0.999), weight_decay=0)

    dataset_type = get_dataset(config)
    '''
    train_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        drop_last=True,
        pin_memory=config.PIN_MEMORY)
    '''
    val_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=True),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        drop_last=True,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    print(config.TRAIN.END_EPOCH)
    for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):
        #function.train(config, train_loader, model, model_D, criterion, optimizer_g, optimizer_d, epoch, writer_dict,log_dir = final_output_dir)
        function.validate(config, val_loader, model, model_D, criterion, epoch, writer_dict,log_dir = final_output_dir)
        adjust_learning_rate(5e-05, optimizer_g, (epoch + 1))
        adjust_learning_rate(2e-05, optimizer_d, (epoch + 1))
        if epoch % args.save_interval == 0:
            save_load_model_generator(model, (epoch + 1), config, final_output_dir)
            save_load_model_discriminator(model_D, (epoch + 1), config, final_output_dir)
    writer_dict['writer'].close()

if __name__ == '__main__':
    main()
    print("over")








