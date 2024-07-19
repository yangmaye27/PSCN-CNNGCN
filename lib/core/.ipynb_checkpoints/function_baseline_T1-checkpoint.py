from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import random
import os

logger = logging.getLogger(__name__)
L1Loss = nn.L1Loss()
from datetime import datetime

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def tensor_to_np_notmask(tensor):
    img = (tensor + 1) / 2
    img = img.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


def mask_gen(tensor, hole_sizemin, hole_sizemmax, pool_w):
    Tensor_mask = torch.cuda.FloatTensor
    mask_in = Tensor_mask(np.zeros((tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])))
    for i in range(tensor.shape[0]):
        n_holes = 1
        for _ in range(n_holes):


            hole_w = random.randint(hole_sizemin, hole_sizemmax)
            hole_w_div2 = hole_w / 2

            hole_h = random.randint(hole_sizemin, hole_sizemmax)
            hole_h_div2 = hole_h / 2

            PoolC_x = random.randint((tensor.shape[2] / 2 - pool_w / 2), (tensor.shape[2] / 2 + pool_w / 2))
            PoolC_y = random.randint((tensor.shape[3] / 2 - pool_w / 2), (tensor.shape[3] / 2 + pool_w / 2))
            mask_in[i, :, int(PoolC_x - hole_w_div2): int(PoolC_x + hole_w_div2),int(PoolC_y - hole_h_div2): int(PoolC_y + hole_h_div2)] = 1.0

    return mask_in


def train(config, train_loader, model, model_D, critertion, optimizer_g, optimizer_d,
          epoch, writer_dict,log_dir = ""):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_ST = AverageMeter()
    loss_M_ST = AverageMeter()

    loss_G = AverageMeter()
    loss_Dis = AverageMeter()

    Fake_Scaler_Log = AverageMeter()
    True_Scaler_Log = AverageMeter()

    model.train()
    model_D.train()



    end = time.time()


    hole_sizemin = 64
    hole_sizemmax = 96
    pool_w = 64
    print(len(train_loader))
   # Takebatchnumber = int(random.randint(3, len(train_loader)) - 2)
    Tensor = torch.cuda.FloatTensor

    for i, (inp_RN, inp_Quant, inp_Vis, inp_guid, inp_Nd, target_256, target_64_jc_all, target_64_st, Slopmap) in enumerate(
            train_loader):
        target_256 = target_256.cuda()
        data_time.update(time.time() - end)

        msks = mask_gen(target_256, hole_sizemin, hole_sizemmax, pool_w)


        inp_RN = inp_RN.cuda(non_blocking=True)
        inp_Vis = inp_Vis.cuda(non_blocking=True)
        inp_Quant = inp_Quant.cuda(non_blocking=True)
        inp_guid = inp_guid.cuda(non_blocking=True)
        inp_Nd = inp_Nd.cuda(non_blocking=True)


        inp_RN = inp_RN * (1 - msks)

        RN_masked = inp_RN.clone()
        inp_msk = msks.clone() * 2 - 1


        optimizer_d.zero_grad()

        output_ST = model(inp_RN, inp_guid, inp_Nd, inp_msk,msks,inp_Vis,inp_Quant)
        valid = Tensor(np.ones((inp_RN.shape[0], 1, 256 // 32, 256 // 32)))
        fake = Tensor(np.zeros((inp_RN.shape[0], 1, 256 // 32, 256 // 32)))
        zero = Tensor(np.zeros((inp_RN.shape[0], 1, 256 // 32, 256 // 32)))

        masks_0_1 = 1 - msks.clone()
        masks_viewed = msks.view(msks.size(0), -1)
        ST_Out_Masked = output_ST * msks+target_256*masks_0_1

        fake_scalar = model_D(ST_Out_Masked.detach(), inp_msk, inp_guid, inp_Nd, RN_masked,inp_Vis,inp_Quant)
        true_scalar = model_D(target_256, inp_msk, inp_guid, inp_Nd, RN_masked,inp_Vis,inp_Quant)

        Fake4log = fake_scalar.clone()
        True4log = true_scalar.clone()

        loss_fake = -torch.mean(torch.min(zero, -valid - fake_scalar))
        loss_true = -torch.mean(torch.min(zero, -valid + true_scalar))

        loss_D = 0.5 * (loss_fake + loss_true)

        loss_D.backward()
        optimizer_d.step()
        loss_Dis.update(loss_D.item(), inp_RN.size(0))

        True_Scaler_Log.update(torch.mean(True4log), inp_RN.size(0))
        Fake_Scaler_Log.update(torch.mean(Fake4log), inp_RN.size(0))

        optimizer_g.zero_grad()

        second_MaskL1Loss = torch.mean(
            torch.abs(ST_Out_Masked - target_256) / masks_viewed.mean(1).view(-1, 1, 1, 1))

        fake_scalar_2 = model_D(ST_Out_Masked, msks, inp_guid, inp_Nd, RN_masked,inp_Vis,inp_Quant)
        GAN_Loss = - torch.mean(fake_scalar_2)


        loss =  50 * second_MaskL1Loss + GAN_Loss
        loss.backward()
        optimizer_g.step()


        losses.update(loss.item(), inp_RN.size(0))
        loss_ST.update(second_MaskL1Loss.item(), inp_RN.size(0))
        loss_M_ST.update(second_MaskL1Loss.item(), inp_RN.size(0))
        loss_G.update(GAN_Loss.item(), inp_RN.size(0))


        batch_time.update(time.time() - end)
        print(i)
        if i % 498 == 0:
            Takeitem = int(random.randint(1, int(inp_RN.size(0))) - 1)
            imgss = []

            score_map_in_rn = inp_RN[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            score_map_in_rn = (score_map_in_rn + 1) / 2 * 255
            score_map_in_guid = inp_guid[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            score_map_in_guid = (score_map_in_guid + 1) / 2 * 255

            score_map_IN = np.zeros((1, 3, 256, 256))
            score_map_IN[:, 0:1, :, :] = score_map_in_rn[:, 0:1, :, :].numpy()
            score_map_IN[:, 1:2, :, :] = score_map_in_guid[:, 0:1, :, :].numpy()
            score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))


            score_map_out = ST_Out_Masked[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            score_map_out = (score_map_out + 1) / 2 * 255
            score_map_IN = np.zeros((1, 3, 256, 256))
            score_map_IN[:, 0:1, :, :] = score_map_out[:, 0:1, :, :].numpy()
            score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

            score_map_GT = target_256[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            score_map_GT = (score_map_GT + 1) / 2 * 255
            score_map_IN = np.zeros((1, 3, 256, 256))
            score_map_IN[:, 0:1, :, :] = score_map_GT[:, 0:1, :, :].numpy()
            score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

            coumtcount = 0
            for names in ['Train_In', 'Train_Out', 'Train_GT']:

                output_dir = log_dir+'/out_imgs/Train/'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                outputname = log_dir+'/out_imgs/Train/Geostreet_' + str(epoch) + "_Train_" + str(i) + "_" + names + "_.png"
                imgss[coumtcount].save(outputname)
                coumtcount = coumtcount + 1


        if i % config.PRINT_FREQ == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            msg = 'T1 Now:  {} Train Epoch {} {}/{},time:{:.4f} Tloss:{:.4f} STloss:{:.4f} ST_M_loss:{:.4f} GANloss:{:.4f} Disloss:{:.4f} FScaler:{:.4f} TScaler:{:.4f} LR G {}, LR D {}'.format(
                current_time, epoch, i, len(train_loader),batch_time.avg, losses.avg, loss_ST.avg, loss_M_ST.avg, loss_G.avg, loss_Dis.avg,
                True_Scaler_Log.avg, Fake_Scaler_Log.avg, optimizer_g.param_groups[0]['lr'], optimizer_d.param_groups[0]['lr'])
            logger.info(msg)
            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    msg = 'T1 Train Epoch {} time:{:.4f} Tloss:{:.4f} JCloss:{:.4f} JC_M_loss:{:.4f} GANloss:{:.4f} Disloss:{:.4f}'.format(
        epoch, batch_time.avg, losses.avg, loss_ST.avg, loss_M_ST.avg, loss_G.avg, loss_Dis.avg)
    logger.info(msg)


def validate(config, val_loader, model, model_D, critertion, epoch, writer_dict=None,log_dir = ""):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    Test_losses = AverageMeter()
    Test_loss_ST = AverageMeter()
    Test_loss_M_ST = AverageMeter()
    Test_loss_G = AverageMeter()
    Test_loss_Dis = AverageMeter()

    model.eval()
    model_D.eval()

    end = time.time()
    lossoutsum = 0

    hole_sizemin = 64
    hole_sizemmax = 96
    pool_w = 64

    Tensor = torch.cuda.FloatTensor

    with torch.no_grad():


        #Takebatchnumber = int(random.randint(3, len(val_loader)) - 2,1)


        for i, (inp_RN, inp_Quant, inp_Vis, inp_guid, inp_Nd, target_256, target_64_jc_all, target_64_st, Slopmap) in enumerate(
                val_loader):
            data_time.update(time.time() - end)
            target_256 = target_256.cuda()

            msks = mask_gen(target_256, hole_sizemin, hole_sizemmax, pool_w)

            inp_RN = inp_RN.cuda(non_blocking=True)
            inp_Vis = inp_Vis.cuda(non_blocking=True)
            inp_Quant = inp_Quant.cuda(non_blocking=True)

            inp_guid = inp_guid.cuda(non_blocking=True)
            inp_Nd = inp_Nd.cuda(non_blocking=True)

            inp_RN = inp_RN * (1 - msks)

            RN_masked = inp_RN.clone()
            inp_msk = msks.clone() * 2 - 1
            output_ST = model(inp_RN, inp_guid, inp_Nd, inp_msk,msks,inp_Vis,inp_Quant)
            masks_0_1 = 1 - msks.clone()

            masks_viewed = msks.view(msks.size(0), -1)

            ST_Out_Masked = output_ST * (msks)+target_256*masks_0_1
            second_MaskL1Loss = torch.mean(
                torch.abs(ST_Out_Masked - target_256) / masks_viewed.mean(1).view(-1, 1, 1, 1))
            fake_scalar_2 = model_D(ST_Out_Masked.detach(), inp_msk, inp_guid, inp_Nd, RN_masked,inp_Vis,inp_Quant)

            GAN_Loss = - torch.mean(fake_scalar_2)
            loss =  50 * second_MaskL1Loss + 1 * GAN_Loss


            Test_losses.update(loss.item(), inp_RN.size(0))
            Test_loss_ST.update(second_MaskL1Loss.item(), inp_RN.size(0))
            Test_loss_M_ST.update(second_MaskL1Loss.item(), inp_RN.size(0))
            Test_loss_G.update(GAN_Loss.item(), inp_RN.size(0))


            batch_time.update(time.time() - end)
            end = time.time()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    msg = 'Now: {} Test Epoch {} time:{:.4f} Tloss:{:.4f} STloss:{:.4f} ST_M_loss:{:.4f} GANloss:{:.4f} Disloss:{:.4f}'.format(
        current_time, epoch, batch_time.avg, Test_losses.avg, Test_loss_ST.avg, Test_loss_M_ST.avg, Test_loss_G.avg,
        Test_loss_Dis.avg)
    print(msg)
    logger.info(msg)

    if writer_dict is not None:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', Test_losses.avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return lossoutsum




