import os
import logging
import torchvision
from .networks import *

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


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
        print("baseline T1")
        extra = config.MODEL.EXTRA
        super(Baseline_Net, self).__init__()
        self.endec_model = InpaintSANet()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(int(7), int(4), 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(int(4), int(7), 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.sa = SpatialAttention()
        self.relu = nn.ReLU()

    def forward(self, x_rn, x_guid, x_Nd, x_msk, msk10, x_Vis, x_Quant):
        Raw_In = torch.cat((x_Quant.clone(), x_Vis.clone()), 1)
        x_out_1 = self.endec_model(masks=msk10, ContextRN=x_rn, GuidRefine=x_guid, NDRefine=x_Nd,
                                   TopoRaw=Raw_In, flag_T1=True)

        return x_out_1

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
        self.block_merge_1 = Conv2dLayer(64, 32, 3, 1, 1, pad_type='zero', activation='lrelu', norm='in', sn=True)
        self.block_merge_2 = Conv2dLayer(32, 1, 3, 1, 1, pad_type='zero', activation='none', norm='none', sn=True)

    def forward(self, output_JC, msks_64, inp_guid, inp_Nd, masked_RN, x_Vis, x_Quant):
        x_In = torch.cat((output_JC, msks_64, inp_guid, inp_Nd, masked_RN, x_Vis, x_Quant), 1)
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

