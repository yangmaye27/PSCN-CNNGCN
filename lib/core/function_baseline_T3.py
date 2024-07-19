
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

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


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

def mask_gen_Test(tensor, pool_w):
    mask_in = torch.tensor(np.zeros((tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])), dtype=torch.float32,
                           device='cuda')
    mask_in_2 = torch.tensor(np.zeros((tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])), dtype=torch.float32,
                             device='cuda')
    mask_in_3 = torch.tensor(np.ones((tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])), dtype=torch.float32,
                             device='cuda')

    for i in range(tensor.shape[0]):
        # 中心点
        PoolC_x = tensor.shape[2] // 2
        PoolC_y = tensor.shape[3] // 2

        # 掩码范围
        hole_w_div2 = 96 // 2
        hole_h_div2 = 96 // 2

        # 设置掩码
        mask_in[i, :, int(PoolC_x - hole_w_div2): int(PoolC_x + hole_w_div2),
        int(PoolC_y - hole_h_div2): int(PoolC_y + hole_h_div2)] = 1.0
        mask_in_2[i, :, int(PoolC_x - hole_w_div2 - 10): int(PoolC_x + hole_w_div2 + 10),
        int(PoolC_y - hole_h_div2 - 10): int(PoolC_y + hole_h_div2 + 10)] = 1.0
        mask_in_3[i, :, int(10): int(246), int(10): int(246)] = 0
        mask_in_2[i, :, :, :] = mask_in_2[i, :, :, :] - mask_in[i, :, :, :] + mask_in_3[i, :, :, :]

    return mask_in, mask_in_2


def mask_gen(tensor, hole_sizemin, hole_sizemmax, pool_w):
    mask_in = torch.tensor(np.zeros((tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])), dtype=torch.float32,
                           device='cuda')
    mask_in_2 = torch.tensor(np.zeros((tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])), dtype=torch.float32,
                             device='cuda')
    mask_in_3 = torch.tensor(np.ones((tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])), dtype=torch.float32,
                             device='cuda')
    for i in range(tensor.shape[0]):
        n_holes = 1

        for _ in range(n_holes):
            
            hole_w = random.randint(hole_sizemin, hole_sizemmax)
            hole_w_div2 = hole_w / 2
            hole_h = random.randint(hole_sizemin, hole_sizemmax)
            hole_h_div2 = hole_h / 2
            PoolC_x = random.randint((tensor.shape[2] / 2 - pool_w / 2), (tensor.shape[2] / 2 + pool_w / 2))
            PoolC_y = random.randint((tensor.shape[3] / 2 - pool_w / 2), (tensor.shape[3] / 2 + pool_w / 2))
            mask_in[i, :, int(PoolC_x - hole_w_div2): int(PoolC_x + hole_w_div2),
            int(PoolC_y - hole_h_div2): int(PoolC_y + hole_h_div2)] = 1.0
            mask_in_2[i, :, int(PoolC_x - hole_w_div2-10): int(PoolC_x + hole_w_div2+10),
            int(PoolC_y - hole_h_div2-10): int(PoolC_y + hole_h_div2+10)] = 1.0
            mask_in_3[i, :, int(10): int(245),int(10): int(245)] = 0
            mask_in_2[i,:,:,:]=mask_in_2[i,:,:,:]-mask_in[i,:,:,:]+mask_in_3[i,:,:,:]

    return mask_in,mask_in_2

def train(config, train_loader, model, model_D, critertion, optimizer_g, optimizer_d,
          epoch, writer_dict,log_dir = ""):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_ST = AverageMeter()
    loss_C_ST = AverageMeter()
    loss_V_ST = AverageMeter()
    loss_G = AverageMeter()
    loss_Dis = AverageMeter()

    Fake_Scaler_Log = AverageMeter()
    True_Scaler_Log = AverageMeter()

    model.train()
    model_D.train()

    end = time.time()

    hole_sizemin = 96
    hole_sizemmax = 96
    pool_w = 64

    Tensor = torch.cuda.FloatTensor

    for i, (inp_RN, inp_Quant, inp_Vis, inp_guid_ss, inp_Nd, target_256, target_64_jc_all, target_64_st, Slopmap,graph_batch) in enumerate(train_loader):


        data_time.update(time.time() - end)
        target_256 = target_256.cuda()
        target_64_jc_all = target_64_jc_all.cuda()

        msks, msk4vt = mask_gen(target_256, hole_sizemin, hole_sizemmax, pool_w)

        inp_Vis = inp_Vis.cuda(non_blocking=True)
        inp_Quant = inp_Quant.cuda(non_blocking=True)
        inp_Nd = inp_Nd.cuda(non_blocking=True)
        inp_RN = inp_RN.cuda(non_blocking=True)
        inp_guid_ss = inp_guid_ss.cuda(non_blocking=True)
        inp_guid = inp_RN * msk4vt + inp_guid_ss * (1 - msk4vt)
        inp_guid = inp_guid.cuda(non_blocking=True)
        # graph_batch = graph_batch.cuda(non_blocking=True)

        # nodes = graph_batch.x
        # edge_index = graph_batch.edge_index
        # edge_attr = graph_batch.edge_attr
        # node_masks=graph_batch.node_masks
        # edge_masks=graph_batch.edge_masks

        inp_RN = inp_RN * (1 - msks)
        RN_masked = inp_RN.clone()
        inp_msk = msks.clone() * 2 - 1

        optimizer_d.zero_grad()

        Output_coarse, Output_vertices, Output_Refine, Topo_out, Topomask_out4vis, pixel_mask, x_att_score = model(
            inp_RN, inp_guid, inp_Nd,
            inp_msk, msks, inp_Vis,
            inp_Quant,graph_batch)

        min_pixel = 0.41
        max_pixel = 0.558
        mask_weight = (pixel_mask - min_pixel) / (max_pixel - min_pixel)
        mask_output = mask_weight * 255
        x_att_score_out = (x_att_score * mask_output).int()

        valid = torch.tensor(np.ones((inp_RN.shape[0], 1, 256 // 32, 256 // 32)), dtype=torch.float32, device='cuda')
        fake = torch.tensor(np.zeros((inp_RN.shape[0], 1, 256 // 32, 256 // 32)), dtype=torch.float32, device='cuda')
        zero = torch.tensor(np.zeros((inp_RN.shape[0], 1, 256 // 32, 256 // 32)), dtype=torch.float32, device='cuda')

        masks_0_1 = 1 - msks.clone()
        masks_viewed = msks.view(msks.size(0), -1)

        ST_Out_Masked = Output_Refine * msks+target_256*masks_0_1
        out_coarse_Masked = Output_coarse * msks+target_256*masks_0_1
        Vt_Out = Output_vertices

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

#         重建损失
        first=torch.mean(torch.abs(ST_Out_Masked - target_256) / masks_viewed.mean(1).view(-1, 1, 1, 1))
        second=torch.mean(torch.abs(out_coarse_Masked - target_256) / masks_viewed.mean(1).view(-1, 1, 1, 1))
        third= torch.mean(torch.abs(Vt_Out - target_256) / masks_viewed.mean(1).view(-1, 1, 1, 1))


        fake_scalar_2 = model_D(ST_Out_Masked.detach(), inp_msk, inp_guid, inp_Nd, RN_masked, inp_Vis, inp_Quant)
        GAN_Loss = - torch.mean(fake_scalar_2)
        loss = 50 * first + 50 * second + 50*third+1 * GAN_Loss
        loss.backward()
        optimizer_g.step()

        losses.update(loss.item(), inp_RN.size(0))
        loss_ST.update(first.item(), inp_RN.size(0))
        loss_C_ST.update(second.item(), inp_RN.size(0))
        loss_V_ST.update(third.item(),inp_RN.size(0))
        loss_G.update(GAN_Loss.item(), inp_RN.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            Takeitem = 0
            imgss = []
            score_map_in_rn = inp_RN[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            score_map_in_rn = (score_map_in_rn + 1) / 2 * 255

            score_map_in_guid = inp_guid[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            score_map_in_guid = (score_map_in_guid + 1) / 2 * 255

            # TEST_IN
            score_map_IN = np.zeros((1, 3, 256, 256))
            score_map_IN[:, 0:1, :, :] = score_map_in_rn[:, 0:1, :, :].numpy()
            score_map_IN[:, 1:2, :, :] = score_map_in_guid[:, 0:1, :, :].numpy()
            score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

            # # Quant_P1
            # score_map_in_quant = inp_Quant[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            # score_map_in_quant = (score_map_in_quant + 1) / 2 * 255
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_quant[:, 0:1, :, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            # # Quant_P2
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_quant[:, 1:2:, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            # # Quant_P3
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_quant[:, 2:3, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            #
            # score_map_in_quant = x_att_score_out[Takeitem:Takeitem + 1, 0:3, :, :].clone().data.cpu()
            #
            # # Quant_P1_mask
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_quant[:, 0:1, :, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            # # Quant_P2_mask
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_quant[:, 1:2:, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            # # Quant_P3_mask
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_quant[:, 2:3, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            #
            # score_map_in_vis = inp_Vis[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            # # Vis_P1
            # score_map_in_vis = (score_map_in_vis + 1) / 2 * 255
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 0:1, :, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            # # Vis_P2
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 1:2, :, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            # # Vis_P3
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 2:3, :, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            # # Vis_P4
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 3:4, :, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            #
            # score_map_in_vis = x_att_score_out[Takeitem:Takeitem + 1, 3:, :, :].clone().data.cpu()
            #
            # # Vis_P1_mask
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 0:1, :, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            # # Vis_P3_mask
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 1:2, :, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            # # Vis_P3_mask
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 2:3, :, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            # # Vis_P4_mask
            # score_map_IN = np.zeros((1, 3, 256, 256))
            # score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 3:4, :, :].numpy()
            # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

            # Ours_coarse
            score_map_out = ST_Out_Masked[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            score_map_out = (score_map_out + 1) / 2 * 255
            score_map_IN = np.zeros((1, 3, 256, 256))
            score_map_IN[:, 0:1, :, :] = score_map_out[:, 0:1, :, :].numpy()
            score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            # ours_refine
            score_map_out = out_coarse_Masked[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            score_map_out = (score_map_out + 1) / 2 * 255
            score_map_IN = np.zeros((1, 3, 256, 256))
            score_map_IN[:, 0:1, :, :] = score_map_out[:, 0:1, :, :].numpy()
            score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))
            #

            score_map_GT = target_256[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            score_map_GT = (score_map_GT + 1) / 2 * 255
            score_map_IN = np.zeros((1, 3, 256, 256))
            score_map_IN[:, 0:1, :, :] = score_map_GT[:, 0:1, :, :].numpy()
            score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

            print_vt = Vt_Out[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            print_vt = (print_vt + 1) / 2 * 255
            score_map_IN = np.zeros((1, 3, 256, 256))
            score_map_IN[:, 0:1, :, :] = print_vt[:, 0:1, :, :].numpy()
            score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

            print_vt_gt = target_64_jc_all[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            print_vt_gt = (print_vt_gt + 1) / 2 * 255
            score_map_IN = np.zeros((1, 3, 256, 256))
            score_map_IN[:, 0:1, :, :] = print_vt_gt[:, 0:1, :, :].numpy()
            score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

            print_mask = mask_output[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()

            score_map_IN = np.zeros((1, 3, 256, 256))
            score_map_IN[:, 0:1, :, :] = print_mask[:, 0:1, :, :].numpy()
            score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

            print_RN = inp_RN[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            print_RN = (print_RN + 1) / 2 * 255
            print_ss = inp_guid_ss[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
            print_ss = (print_ss + 1) / 2 * 255
            score_map_IN = np.zeros((1, 3, 256, 256))
            score_map_IN[:, 0:1, :, :] = print_RN[:, 0:1, :, :].numpy()
            score_map_IN[:, 1:2, :, :] = print_ss[:, 0:1, :, :].numpy()
            score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
            imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

            coumtcount = 0
            # for names in ['Test_In', 'Quant_P1', 'Quant_P2', 'Quant_P3',
            #               'Quant_P1_mask', 'Quant_P2_mask', 'Quant_P3_mask',
            #               'Vis_P1', 'Vis_P2', 'Vis_P3', 'Vis_P4',
            #               'Vis_P1_mask', 'Vis_P2_mask', 'Vis_P3_mask', 'Vis_P4_mask',
            #               'ours_coarse', 'ours_refine',
            #               'GT', "pred_vt",
            #               "gt_vt", "mask", "rn_guid"]:
            for names in ['Test_In',
                          'ours_coarse', 'ours_refine',
                          'GT', "pred_vt",
                          "gt_vt", "mask", "rn_guid"]:
                output_dir = log_dir + '/out_imgs/Train/'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                outputname = log_dir + '/out_imgs/Train/Geostreet_' + str(epoch) + "_Train_" + str(
                    i * 8) + "_" + names + "_.png"
                print(coumtcount)
                if coumtcount >= len(imgss):
                    break;
                imgss[coumtcount].save(outputname)
                coumtcount = coumtcount + 1
        msg='T3 Train Epoch {} time:{:.4f} Tloss:{:.4f} loss_ST:{:.4f} loss_C_ST:{:.4f} loss_V_ST:{:.4f}  GANloss:{:.4f} Disloss:{:.4f}'.format(
        epoch, batch_time.avg, losses.avg, loss_ST.avg, loss_C_ST.avg, loss_V_ST.avg,loss_G.avg, loss_Dis.avg)
        logger.info(msg)


def validate(config, val_loader, model, model_D, critertion, epoch, writer_dict, log_dir="", base_model=None,
             base_modelT3=None, base_modelT1=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    Test_losses = AverageMeter()
    Test_loss_ST = AverageMeter()
    Test_loss_M_ST = AverageMeter()
    Test_loss_M_ST_baseline = AverageMeter()
    Test_loss_G = AverageMeter()
    Test_loss_Dis = AverageMeter()

    initial_flag = True

    model.eval()
    model_D.eval()
    if base_model is not None:
        base_model.eval()

        base_modelT1.eval()

    end = time.time()
    lossoutsum = 0

    hole_sizemin = 96
    hole_sizemmax = 96
    pool_w = 64
    Tensor = torch.cuda.FloatTensor
    length = len(val_loader)

    count = 0
    total_mae = 0
    total_mse = 0
    total_ssim = 0
    total_psnr=0

    with torch.no_grad():
        # Takebatchnumber = int(random.randint(3, len(val_loader)) - 2)
        coumtcount = 0
        for i, (
                inp_RN, inp_Quant, inp_Vis, inp_guid_ss, inp_Nd, target_256, target_64_jc_all, target_64_st,
                Slopmap, graphBatch) in enumerate(val_loader):
            data_time.update(time.time() - end)
            target_256 = target_256.cuda()
            target_64_jc_all = target_64_jc_all.cuda()

            # msks, msk4vt = mask_gen(target_256, hole_sizemin, hole_sizemmax, pool_w)
            msks, msk4vt = mask_gen_Test(target_256, pool_w)

            inp_Vis = inp_Vis.cuda(non_blocking=True)
            inp_Quant = inp_Quant.cuda(non_blocking=True)

            inp_Nd = inp_Nd.cuda(non_blocking=True)
            inp_RN = inp_RN.cuda(non_blocking=True)
            inp_guid_ss = inp_guid_ss.cuda(non_blocking=True)
            inp_guid = inp_RN * msk4vt + inp_guid_ss * (1 - msk4vt)

            inp_guid = inp_guid.cuda(non_blocking=True)
            inp_RN = inp_RN * (1 - msks)
            RN_masked = inp_RN.clone()

            inp_msk = msks.clone() * 2 - 1
            Output_coarse, Output_vertices, Output_Refine, Topo_out, Topomask_out4vis, pixel_mask, x_att_score = model(
                inp_RN, inp_guid, inp_Nd,
                inp_msk, msks, inp_Vis,
                inp_Quant,graphBatch)
            min_pixel = 0.41
            max_pixel = 0.558
            mask_weight = (pixel_mask - min_pixel) / (max_pixel - min_pixel)
            mask_output = mask_weight * 255
            x_att_score_out = (x_att_score * mask_output).int()
            masks_0_1 = 1 - msks.clone()
            masks_viewed = msks.view(msks.size(0), -1)
            ST_Out_Masked_Coarse = Output_coarse * msks + target_256 * masks_0_1
            ST_Out_Masked_Refine = Output_Refine * msks + target_256 * masks_0_1

            output_ST_baseline = target_256
            output_ST_baseline_T1 = target_256
            coarse_out_T2 = target_256

            output_ST_cpu = Output_Refine.cpu().numpy()
            output_STCorase_cpu = Output_coarse.cpu().numpy()
            target_256_cpu = target_256.cpu().numpy()
            masks_cpu = msks.cpu().numpy()
            masks_0_1_cpu = masks_0_1.cpu().numpy()

            Vt_Out = Output_vertices

            if initial_flag:

                pixel_mask_log = pixel_mask.detach().cpu()
                initial_flag = False
            else:

                pixel_mask_log = torch.cat((pixel_mask_log, pixel_mask.detach().cpu()), dim=0)

            for j in range(12):

                imgss = []
                # score_map_in_rn = inp_RN[j, :, :, :].clone().data.cpu()
                # score_map_in_rn = (score_map_in_rn + 1) / 2 * 255
                #
                # score_map_in_guid = inp_guid[j, :, :, :].clone().data.cpu()
                # score_map_in_guid = (score_map_in_guid + 1) / 2 * 255
                #
                # score_map_IN = np.zeros((1, 3, 256, 256))
                # score_map_IN[0, 0:1, :, :] = score_map_in_rn[ :, :, :].numpy()
                # score_map_IN[0, 1:2, :, :] = score_map_in_guid[:, :, :].numpy()
                # score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                # imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                masks_cpu_Item = masks_cpu[j, :, :, :]
                output_ST_cpu_Item = output_ST_cpu[j, :, :, :]
                masks_0_1_cpu_Item = masks_0_1_cpu[j, :, :, :]
                target_256_cpu_Item = target_256_cpu[j, :, :, :]

                red_mask = ((masks_cpu_Item * output_ST_cpu_Item + masks_0_1_cpu_Item * target_256_cpu_Item) + 1) / 2 * 255
                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[0, 0, :, :] = red_mask  # 设置红色通道
                white_non_mask = ((masks_0_1_cpu_Item * target_256_cpu_Item) + 1 - masks_cpu_Item) / 2 * 255
                score_map_IN[0, 1:2, :, :] = white_non_mask  # 设置绿色和蓝色通道
                score_map_IN[0, 2:3, :, :] = white_non_mask  # 设置绿色和蓝色通道
                img1=red_mask.squeeze(0).astype('uint8')
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))


                # Ground Truth
                score_map_GT = target_256[j, :, :, :].clone().data.cpu()
                score_map_GT = (score_map_GT + 1) / 2 * 255
                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[0, 0:1, :, :] = score_map_GT[ 0:1, :, :].numpy()
                img2=score_map_GT.squeeze(0).numpy().astype('uint8')
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                countOut = ST_Out_Masked_Refine * (msks)
                GT = target_256 * (msks)

                start_x, start_y = 80, 80
                end_x, end_y = 176, 176
                count_loss_GT = img1[start_x:end_x, start_y:end_y]
                count_loss_Out = img2[start_x:end_x, start_y:end_y]

                # numpys_GT=count_loss_GT.cpu().numpy()
                # numpys_Out=count_loss_Out.cpu().numpy()
                # np.set_printoptions(threshold=np.inf)
                # print(numpys_GT)
                # print(numpys_Out)
                # binary_array_GT = np.where(numpys_GT < -0.5, 1, 0)
                # binary_array_OUT = np.where(numpys_Out < -0.5, 1, 0)

                # 计算MAE和MSE
                mae = np.mean(np.abs(count_loss_GT-count_loss_Out))
                mse = np.mean(np.square(count_loss_GT-count_loss_Out))
                # 计算SSIM
                # ssim_value = ssim(count_loss_GT.cpu().numpy(), count_loss_Out.cpu().numpy(),data_range=count_loss_GT.cpu().numpy().max()-count_loss_GT.cpu().numpy().min())

                ssim_value = ssim(count_loss_GT, count_loss_Out)

                psnr_value=psnr(count_loss_GT, count_loss_Out)
                total_mae += mae
                total_mse += mse
                total_ssim += ssim_value
                if(np.isinf(psnr_value)):
                    total_psnr+=100
                else:
                    total_psnr+=psnr_value
                count += 1

                coumtcount=0
                for names in [ '输出','真实']:
                    print(count)
                    output_dir = log_dir + '/out_imgs/'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    outputname = log_dir + '/out_imgs/Geostreet_' + str(epoch) + "_Test_" + str(
                        i) + "_" + names + "_"+str(j)+".png"
                    if coumtcount >= len(imgss):
                        break;
                    imgss[coumtcount].save(outputname)
                    coumtcount = coumtcount + 1

    # save_dir = './out_imgs/all_out.npy'
    average_mae = total_mae / count
    average_mse = total_mse / count
    average_ssim = total_ssim / count
    average_psnr=total_psnr/count

    print(f"平均MAE: {average_mae}")
    print(f"平均MSE: {average_mse}")
    print(f"平均SSIM: {average_ssim}")
    print(f"平均PSNR: {average_psnr}")
    import pickle
    # with open(save_dir, "wb") as fp:
    #     pickle.dump(all_data, fp)
    return lossoutsum




