import os
import logging
import torchvision
from .networks import *

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

# 空间注意力机制代码
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class Baseline_Net(nn.Module):

    def __init__(self, config, **kwargs):
        self.inplanes = 64
        print("baseline ATT")
        super(Baseline_Net, self).__init__()

        self.endec_model = InpaintSANet_v2()
        self.model_hr = HighResolutionNet(config.MODEL.EXTRA)
        # 池化操作通常是使用指定窗口大小的区域中的总体统计特征，代替输入向量在该区域的值，用于降低卷积操作带来的计算参数量
        # 自适应池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 使用leakyReLu激活函数
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(int(7), int(4), 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(int(4), int(7), 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.sa = SpatialAttention()
        self.relu = nn.ReLU()

    # 传入路网结构
    def forward(self, x_rn, x_guid, x_Nd, x_msk, msk10, x_Vis, x_Quant,graph_batch):

        # 地理信息通道融合
        Raw_In = torch.cat((x_Quant.clone(), x_Vis.clone()), 1)
        # GE模块
        Raw_In_atted, pixel_mask, x_att_score = self.model_hr(Raw_In)

        pixel_mask_4vis = pixel_mask.clone()

        x_out_1, x_out_vertices, x_out_2, Topoprocessed, pixel_mask = self.endec_model(masks=msk10, ContextRN=x_rn,
                                                                           GuidRefine=x_guid, NDRefine=x_Nd,
                                                                           TopoRaw=Raw_In_atted,
                                                                           pixel_mask=pixel_mask,
                                                                                       graph_batch=graph_batch)

        return x_out_1, x_out_vertices, x_out_2, Topoprocessed, pixel_mask_4vis, pixel_mask, x_att_score

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):

                nn.init.normal_(m.weight, std=0.001)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()


        self.block1 = Conv2dLayer(12, 64, 7, 1, 3, pad_type='zero', activation='lrelu', norm='none', sn=True)
        self.block2 = Conv2dLayer(64, 64 * 2, 4, 2, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block3 = Conv2dLayer(64 * 2, 64 * 4, 4, 2, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block4 = Conv2dLayer(64 * 4, 64 * 4, 4, 2, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block5 = Conv2dLayer(64 * 4, 64 * 4, 4, 2, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block6 = Conv2dLayer(64 * 4, 64, 4, 2, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        """
        self.block1_JC = Conv2dLayer(12, 64, 7, 1, 3, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = True)
        self.block2_JC = Conv2dLayer(64, 64 * 2, 4, 2, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in', sn = True)
        self.block3_JC = Conv2dLayer(64 * 2, 64 * 4, 4, 2, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in', sn = True)
        
        
        self.block6_JC = Conv2dLayer(64 * 4, 1, 4, 2, 1, pad_type = 'zero', activation = 'none', norm = 'none', sn = True)

        self.block1_ST = Conv2dLayer(12, 64, 7, 1, 3, pad_type='zero', activation='lrelu', norm='none', sn=True)
        self.block2_ST = Conv2dLayer(64, 64 * 2, 4, 2, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block3_ST = Conv2dLayer(64 * 2, 64 * 4, 4, 2, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block6_ST = Conv2dLayer(64 * 4, 1, 4, 2, 1, pad_type='zero', activation='none', norm='none', sn=True)
        """

        self.block_merge_1 = Conv2dLayer(64, 32, 3, 1, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block_merge_2 = Conv2dLayer(32, 1, 3, 1, 1, pad_type='zero', activation='none', norm='none', sn=True)

    def forward(self, output_JC, msks_64, inp_guid, inp_Nd, masked_RN,inp_Vis,inp_Quant):

        x_In = torch.cat((output_JC, msks_64, inp_guid, inp_Nd, masked_RN,inp_Vis,inp_Quant), 1)
        x_In = self.block1(x_In)
        x_In = self.block2(x_In)
        x_In = self.block3(x_In)
        x_In = self.block4(x_In)
        x_In = self.block5(x_In)
        x_In = self.block6(x_In)
        x_In = self.block_merge_1(x_In)
        x_In = self.block_merge_2(x_In)
        return x_In


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        block = [torchvision.models.vgg16(pretrained=True).features[:15].eval()]
        for p in block[0]:
            p.requires_grad = False
        self.block = torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        for block in self.block:
            x = block(x)
        return x


def weights_init(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


    net.apply(init_func)


def get_baseline_net(config, **kwargs):
    model = Baseline_Net(config, **kwargs)
    pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
    weights_init(net=model, init_type='xavier', init_gain=0.02)
    return model


def D_net():
    model = DNet()
    weights_init(net=model, init_type='xavier', init_gain=0.02)
    return model


def P_net():
    model = PNet()
    return model

