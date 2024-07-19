





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
    mask_in_2 = Tensor_mask(np.zeros((tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])))
    mask_in_3 = Tensor_mask(np.ones((tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])))
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


def validate(config, val_loader, model, model_D, critertion, epoch, writer_dict, log_dir="", base_model = None, base_modelT3 = None,  base_modelT1 = None):
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

    hole_sizemin = 64
    hole_sizemmax = 96
    pool_w = 64
    Tensor = torch.cuda.FloatTensor
    length = len(val_loader)
    with torch.no_grad():
        Takebatchnumber = int(random.randint(3, len(val_loader)) - 2)
        
        
        
        
        for i, (
                inp_RN, inp_Quant, inp_Vis, inp_guid_ss, inp_Nd, target_256, target_64_jc_all, target_64_st,
                Slopmap) in enumerate(
            val_loader):
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
            inp_RN = inp_RN * (1 - msks)
            RN_masked = inp_RN.clone()
            
            
            inp_msk = msks.clone() * 2 - 1
            Output_coarse, Output_vertices, Output_Refine, Topo_out, Topomask_out4vis, pixel_mask, x_att_score = model(inp_RN, inp_guid, inp_Nd,
                                                                                              inp_msk, msks, inp_Vis,
                                                                                              inp_Quant)
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
            
            
            
            
            
            
            
            
            
            
            
            
            


            Vt_Out = Output_vertices
            
            
            
            
            
            
            
            
            
            
            
            

            
            if initial_flag:
                
                
                
                
                
                
                
                
                
                pixel_mask_log = pixel_mask.detach().cpu()
                initial_flag = False
            else:
                
                
                
                
                
                
                
                
                
                pixel_mask_log = torch.cat((pixel_mask_log,pixel_mask.detach().cpu()),dim=0)
            
            
            if True:
                
                Takeitem = 0
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

                score_map_in_quant = inp_Quant[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
                score_map_in_quant = (score_map_in_quant + 1) / 2 * 255
                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_quant[:, 0:1, :, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_quant[:, 1:2:, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_quant[:, 2:3, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                
                score_map_in_quant = x_att_score_out[Takeitem:Takeitem + 1, 0:3, :, :].clone().data.cpu()
                
                
                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_quant[:, 0:1, :, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_quant[:, 1:2:, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_quant[:, 2:3, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                score_map_in_vis = inp_Vis[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
                
                score_map_in_vis = (score_map_in_vis + 1) / 2 * 255
                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 0:1, :, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 1:2, :, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 2:3, :, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 3:4, :, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                
                score_map_in_vis = x_att_score_out[Takeitem:Takeitem + 1, 3:, :, :].clone().data.cpu()
                
                
                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 0:1, :, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 1:2, :, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 2:3, :, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_in_vis[:, 3:4, :, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                

                score_map_out = ST_Out_Masked_Coarse[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
                score_map_out = (score_map_out + 1) / 2 * 255
                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_out[:, 0:1, :, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                score_map_out = ST_Out_Masked_Refine[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
                score_map_out = (score_map_out + 1) / 2 * 255
                score_map_IN = np.zeros((1, 3, 256, 256))
                score_map_IN[:, 0:1, :, :] = score_map_out[:, 0:1, :, :].numpy()
                score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                if base_model is not None:
                    score_map_out = output_ST_baseline[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
                    score_map_out = (score_map_out + 1) / 2 * 255
                    score_map_IN = np.zeros((1, 3, 256, 256))
                    score_map_IN[:, 0:1, :, :] = score_map_out[:, 0:1, :, :].numpy()
                    score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                    imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                    score_map_out = coarse_out_T2[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
                    score_map_out = (score_map_out + 1) / 2 * 255
                    score_map_IN = np.zeros((1, 3, 256, 256))
                    score_map_IN[:, 0:1, :, :] = score_map_out[:, 0:1, :, :].numpy()
                    score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                    imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                    score_map_out = output_ST_baseline_T1[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
                    score_map_out = (score_map_out + 1) / 2 * 255
                    score_map_IN = np.zeros((1, 3, 256, 256))
                    score_map_IN[:, 0:1, :, :] = score_map_out[:, 0:1, :, :].numpy()
                    score_map_IN = score_map_IN.squeeze(0).transpose((1, 2, 0))
                    imgss.append(Image.fromarray(score_map_IN.astype('uint8')).convert('RGB'))

                    
                    
                    
                    
                    
                    
                else:
                    score_map_out = ST_Out_Masked_Refine[Takeitem:Takeitem + 1, :, :, :].clone().data.cpu()
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
                for names in ['Test_In', 'Quant_P1', 'Quant_P2', 'Quant_P3',
                              'Quant_P1_mask', 'Quant_P2_mask', 'Quant_P3_mask',
                              'Vis_P1','Vis_P2', 'Vis_P3', 'Vis_P4',
                              'Vis_P1_mask', 'Vis_P2_mask', 'Vis_P3_mask', 'Vis_P4_mask',
                              'ours_coarse','ours_refine', "T2_refine",
                              "T2_coarse", "T1_refine",
                              'GT', "pred_vt",
                              "gt_vt", "mask","rn_guid"]:
                    output_dir = log_dir + '/out_imgs/Test/'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    outputname = log_dir + '/out_imgs/Test/Geostreet_' + str(epoch) + "_Test_" + str(
                        i*8) + "_" + names + "_.png"
                    print(coumtcount)
                    if coumtcount>= len(imgss):
                        break;
                    imgss[coumtcount].save(outputname)
                    coumtcount = coumtcount + 1

            
            
            
            
            
            

            
            
            
            if i % 5 == 0:
                now = datetime.now()
                print(i, "/", length, "0604 now:", now)
                

    
    
    
    

    
    
    
    
    

    all_data = {
    
    
    
    
    
    
    
    
    
    "pixel_mask_log" : pixel_mask_log.numpy().astype(np.float16)
    }



    save_dir = './out_imgs/all_out.npy'
    import pickle
    with open(save_dir, "wb") as fp:  
        pickle.dump(all_data, fp)
    return lossoutsum, all_data





