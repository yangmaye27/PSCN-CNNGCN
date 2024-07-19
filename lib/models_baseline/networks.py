import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False):
        super(Conv2dLayer, self).__init__()

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation, with_attn=False):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        if self.with_attn:
            return out, attention
        else:
            return out


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


class GatedConv2dWithActivation(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class GatedDeConv2dWithActivation(torch.nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv2dWithActivation, self).__init__()
        self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='reflect',
                 activation='lrelu', norm='none', sn=False):
        super(GatedConv2d, self).__init__()

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
            self.mask_conv2d = SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):

        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:

            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, norm_layer=None):
        super(HighResolutionModule, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.LeakyReLU(0.1)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3)))
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_outchannels_conv3x3,
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.LeakyReLU(0.1)))
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_outchannels_conv3x3,
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.LeakyReLU(0.1)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=True
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self,
                 cfg,
                 norm_layer=None):
        super(HighResolutionNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.conv_16to64 = nn.Sequential(
            nn.ConvTranspose2d(7, 16, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            self.norm_layer(16),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 16, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            self.norm_layer(16),
            nn.LeakyReLU(0.1)
        )
        self.conv_16 = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            self.norm_layer(32),
            nn.LeakyReLU(0.1)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, stride=2, padding=1,
                      bias=False),
            self.norm_layer(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                      bias=False),
            self.norm_layer(64),
            nn.LeakyReLU(0.1)
        )
        self.change_channel_cnn1 = nn.Conv2d(23, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.change_channel_cnn2 = nn.Conv2d(15, 8, kernel_size=3, stride=1, padding=1, bias=False)

        self.avgpool = nn.AvgPool2d((4, 4))

        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage1_cfg['BLOCK']]

        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.stage1, pre_stage_channels = self._make_stage(
            self.stage1_cfg, num_channels)

        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        last_inp_channels = 120
        self.last_layer_att = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            self.norm_layer(last_inp_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=7,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        self.last_layer_mask = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            self.norm_layer(last_inp_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0)
        )

        self.first_16to64 = self._make_trans_conv(32, 2)
        self.first_4to64 = self._make_trans_conv(64, 4)
        self.first_16to1 = self._make_conv(32, 4)
        self.first_4to1 = self._make_conv(64, 2)

        self.middle_64to256 = self._make_trans_conv(16, 2)
        self.middle_16to256 = self._make_trans_conv(32, 4)
        self.middle_4to256 = self._make_trans_conv(64, 6)
        self.middle_64to1 = self._make_conv(16, 6)
        self.middle_16to1 = self._make_conv(32, 4)
        self.middle_4to1 = self._make_conv(64, 2)

        self.final_256to1 = self._make_conv(8, 8)
        self.final_64to1 = self._make_conv(16, 6)
        self.final_16to1 = self._make_conv(32, 4)
        self.final_4to1 = self._make_conv(64, 2)

        self.att_linear = nn.Sequential(nn.Linear(128 * 3, 48),

                                        nn.Linear(48, last_inp_channels))

    def _make_conv(self, inchannel, n_layer):
        model_list = []
        inchannel_temp = inchannel
        for i in range(n_layer):
            model_list.extend([nn.Conv2d(inchannel, inchannel_temp, kernel_size=3,
                                         stride=2, padding=1),
                               self.norm_layer(inchannel_temp),
                               nn.LeakyReLU(0.1)])
            if i % 2 == 1:
                inchannel = inchannel_temp
            else:
                inchannel = inchannel_temp
                inchannel_temp = inchannel * 2
        return nn.Sequential(*model_list)

    def _make_trans_conv(self, inchannel, n_layer):
        model_list = []
        inchannel_temp = inchannel
        for i in range(n_layer):
            model_list.extend([nn.ConvTranspose2d(inchannel, inchannel_temp, kernel_size=3,
                                                  stride=2, padding=1, output_padding=1),
                               self.norm_layer(inchannel_temp),
                               nn.LeakyReLU(0.1)]

                              )
            if i % 2 == 1:
                inchannel = inchannel_temp
            else:
                inchannel = inchannel_temp
                inchannel_temp = inchannel // 2
        return nn.Sequential(*model_list)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        self.norm_layer(num_channels_cur_layer[i]),
                        nn.LeakyReLU(0.1)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        self.norm_layer(outchannels),
                        nn.LeakyReLU(0.1)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):

            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output,
                                     norm_layer=self.norm_layer)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x_raw = x

        x_64_raw = self.avgpool(x_raw)
        x_16_raw = self.avgpool(x_64_raw)
        x_16 = self.conv_16(x_16_raw)
        x_4 = self.conv_4(x_16_raw)

        x_fused = self.stage1([x_16, x_4])
        out_feat_64_stage1 = self.first_16to64(x_fused[0]) + self.first_4to64(x_fused[1])
        out_feat_1_stage1 = self.first_16to1(x_fused[0]) + self.first_4to1(x_fused[1])
        x_64_input = torch.cat((x_64_raw, out_feat_64_stage1), dim=1)
        x_64_input = self.change_channel_cnn1(x_64_input)

        x_fused = self.stage2([x_64_input, x_fused[0], x_fused[1]])
        out_feat_256_stage2 = self.middle_64to256(x_fused[0]) + self.middle_16to256(x_fused[1]) + self.middle_4to256(
            x_fused[2])

        out_feat_1_stage2 = self.middle_64to1(x_fused[0]) + self.middle_16to1(x_fused[1]) + self.middle_4to1(x_fused[2])
        x_256_input = torch.cat((x_raw, out_feat_256_stage2), dim=1)
        x_256_input = self.change_channel_cnn2(x_256_input)

        x_fused = self.stage3([x_256_input, x_fused[0], x_fused[1], x_fused[2]])
        out_feat_1_stage3 = self.final_256to1(x_fused[0]) + self.final_64to1(x_fused[1]) + \
                            self.final_16to1(x_fused[2]) + self.final_4to1(x_fused[3])

        out_feat_1 = torch.cat((out_feat_1_stage1, out_feat_1_stage2, out_feat_1_stage3), dim=1)
        attention_weight = self.att_linear(out_feat_1.view(out_feat_1.size(0), -1)).sigmoid()

        x0_h, x0_w = 256, 256
        x1 = F.interpolate(x_fused[1], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        x2 = F.interpolate(x_fused[2], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        x3 = F.interpolate(x_fused[3], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        x = torch.cat([x_fused[0], x1, x2, x3], 1)
        x = x * attention_weight.unsqueeze(-1).unsqueeze(-1)
        x_att = self.last_layer_att(x)
        x_mask = self.last_layer_mask(x)

        x_atted = x_raw * x_att.sigmoid()
        pixel_mask = x_mask.sigmoid()

        return x_atted, pixel_mask, x_att.sigmoid()


class InpaintSANet_v2(torch.nn.Module):

    def __init__(self, n_in_channel=11):
        super(InpaintSANet_v2, self).__init__()
        cnum = 32
        print("change encoder")

        self.coarse_net = nn.Sequential(

            GatedConv2dWithActivation(4, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            GatedConv2dWithActivation(cnum, 2 * cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),

            GatedConv2dWithActivation(2 * cnum, 4 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
        )

        self.conv_topo_vt_pred = nn.Sequential(
            GatedConv2dWithActivation(9, cnum, 5, 1, padding=get_pad(256, 5, 1)),

            GatedConv2dWithActivation(cnum, 2 * cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),

            GatedConv2dWithActivation(2 * cnum, 4 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        )

        # self.conv_topo = nn.Sequential(
        #     GatedConv2dWithActivation(9, cnum, 5, 1, padding=get_pad(256, 5, 1)),
        #
        #     GatedConv2dWithActivation(cnum, 2 * cnum, 4, 2, padding=get_pad(256, 4, 2)),
        #     GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
        #
        #     GatedConv2dWithActivation(2 * cnum, 4 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
        #     GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        #     GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        #     GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        # )

        self.coarse_upsample_net = nn.Sequential(
            GatedConv2dWithActivation(8 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedDeConv2dWithActivation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),

            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum // 2, 1, 3, 1, padding=get_pad(128, 3, 1), activation=None)
        )

        self.coarse_topobranch_upsample_net = nn.Sequential(
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedDeConv2dWithActivation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),

            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum // 2, 1, 3, 1, padding=get_pad(128, 3, 1), activation=None)
        )

        # self.refine_conv_net = nn.Sequential(
        #
        #     GatedConv2dWithActivation(5, cnum, 5, 1, padding=get_pad(256, 5, 1)),
        #
        #     GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
        #     GatedConv2dWithActivation(cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
        #
        #     GatedConv2dWithActivation(2 * cnum, 2 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
        #     GatedConv2dWithActivation(2 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        #     GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        #     GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        #     GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
        #     GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
        #     GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
        #     GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        # )
        # 下采样
        self.refine_down1 = nn.Sequential(
            GatedConv2dWithActivation(5, cnum, 5, 1, padding=get_pad(256, 5, 1)),
        )

        self.refine_down2 = nn.Sequential(
            GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
        )

        self.refine_down3 = nn.Sequential(
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(2 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        )

        self.refine_conv_net_CA_branch_prior = nn.Sequential(

            GatedConv2dWithActivation(5, cnum, 5, 1, padding=get_pad(256, 5, 1)),

            GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),

            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(2 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

        )
        self.refine_attn = Self_Attn(4 * cnum, 'relu', with_attn=False)

        self.refine_conv_net_CA_branch_post = nn.Sequential(
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        )

        self.refine_conv_net_topobranch = nn.Sequential(
            GatedConv2dWithActivation(4, cnum, 5, 1, padding=get_pad(256, 5, 1)),

            GatedConv2dWithActivation(cnum, 2 * cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),

            GatedConv2dWithActivation(2 * cnum, 4 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        )

        # self.refine_upsample_net = nn.Sequential(
        #     GatedConv2dWithActivation(12 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        #     GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        #     GatedDeConv2dWithActivation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
        #     GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
        #     GatedDeConv2dWithActivation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),
        #     GatedConv2dWithActivation(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1)),
        #     GatedConv2dWithActivation(cnum // 2, 1, 3, 1, padding=get_pad(256, 3, 1), activation=None),
        # )

        # 上采样3次
        self.refine_up1 = nn.Sequential(
            GatedConv2dWithActivation(16 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedDeConv2dWithActivation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
        )
        self.refine_up2 = nn.Sequential(
            GatedConv2dWithActivation(4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),
        )
        self.refine_up3 = nn.Sequential(
            GatedConv2dWithActivation(2*cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1)),
            GatedConv2dWithActivation(cnum // 2, 1, 3, 1, padding=get_pad(256, 3, 1), activation=None),
        )

        self.lowChannel=GatedConv2dWithActivation(128,10,3,1,1)

        self.lowChannelOne=GatedConv2dWithActivation(128,1,3,1,1)

        self.to64Channel=GatedConv2dWithActivation(128,64,3,1,1)

        self.upChannel=GatedConv2dWithActivation(1,128,3,1,1)

        self.avgpool4 = nn.AvgPool2d((4, 4))

    def forward(self, masks=None, ContextRN=None, GuidRefine=None, NDRefine=None, TopoRaw=None, pixel_mask=None,
                graph_batch=None):

        TopoRaw = TopoRaw * pixel_mask
        topo_processed = TopoRaw.clone()
        x_in = torch.cat((ContextRN, GuidRefine, NDRefine, masks), 1)
        x = self.coarse_net(x_in)
        # print("x的形状为",x.shape())

        x_in_topo_branch_vt = torch.cat((GuidRefine, NDRefine, TopoRaw), 1)
        x_topo_branch_4vertices_in = self.conv_topo_vt_pred(x_in_topo_branch_vt)
        # print("conv_topo的形状为：",x_topo_branch_4vertices_in.shape())

        x_topo_branch = x_topo_branch_4vertices_in.clone()

        x_topo_branch_4vertices = self.coarse_topobranch_upsample_net(x_topo_branch_4vertices_in)
        x_topo_branch_4vertices_out = x_topo_branch_4vertices.tanh().clone()
        x_cont_in = torch.cat((x, x_topo_branch), 1)
        x_cont = self.coarse_upsample_net(x_cont_in).tanh()

        coarse_x = x_cont
        coarse_x_out = coarse_x.clone()

        # 只关注修复区域部分的内容
        masked_imgs = ContextRN * (1 - masks) + coarse_x * masks
        masked_imgs_4CAbranch = masked_imgs.clone()
        masked_imgs_4Topobranch = masked_imgs.clone()
        x_in_refine = torch.cat((masked_imgs, GuidRefine, NDRefine, masks, x_topo_branch_4vertices_out), 1)
        # CM模块

        # 第二次refine Encoder-Decoder结构
        # x_refine = self.refine_conv_net(x_in_refine)
        # 256 * 256
        x_refine_down1 = self.refine_down1(x_in_refine)
        # 128 * 128
        x_refine_down2 = self.refine_down2(x_refine_down1)
        # 64 * 64
        x_refine = self.refine_down3(x_refine_down2)

        # print("x_refine:",x_refine.shape)

        # ca
        x_in_refine_CA_branch = torch.cat(
            (masked_imgs_4CAbranch, GuidRefine, NDRefine, masks, x_topo_branch_4vertices_out), 1)
        x_refine_CA_branch = self.refine_conv_net_CA_branch_prior(x_in_refine_CA_branch)
        x_refine_CA_branch = self.refine_attn(x_refine_CA_branch)
        x_refine_CA_branch = self.refine_conv_net_CA_branch_post(x_refine_CA_branch)

        x_in_refine_topo_branch = torch.cat(
            (GuidRefine, NDRefine, x_topo_branch_4vertices_out, masked_imgs_4Topobranch), 1)
        x_refine_topo_branch = self.refine_conv_net_topobranch(x_in_refine_topo_branch)

        # node_features = graph_batch.x
        # edge_index = graph_batch.edge_index
        # edge_attr = graph_batch.edge_attr
        # batch = graph_batch.batch  # batch 标识每个图的节点属于哪个图
        # node_masks = graph_batch.node_masks
        # edge_masks = graph_batch.edge_masks

        x_refine_input=self.lowChannel(x_refine)


        local_features_list = []
        for graph_id in range(len(graph_batch)):
            graph_node_features = graph_batch[graph_id].x
            nodes_scaled = graph_node_features // 4
            local_features = []
            for node in nodes_scaled:
                nodes_x_64, nodes_y_64 = int(node[0]), int(node[1])
                expanded_feature = torch.zeros(x_refine_input.size(1), 5, 5)
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        new_x, new_y = nodes_x_64 + i, nodes_y_64 + j
                        new_x = max(0, min(new_x, x_refine_input.size(2) - 1))
                        new_y = max(0, min(new_y, x_refine_input.size(3) - 1))
                        tempFeature = x_refine_input[graph_id, :, :, :]
                        for k in range(x_refine_input.size(1)):
                            expanded_feature[k, i + 2, j + 2] = tempFeature[k, new_x, new_y]
                local_features.append(expanded_feature)
            local_features_list.append(torch.stack(local_features))
        # local_features_list = torch.stack(local_features_list)
        new_local_features_list=[]
        for w in local_features_list:
            new_local_features_list.append(w.view(w.shape[0],-1))

        data_list = []
        for i in range(len(graph_batch)):
            data = Data(x=new_local_features_list[i], edge_index=graph_batch[i].edge_index,edge_attr=graph_batch[i].edge_attr,node_masks=graph_batch[i].node_mask,edge_masks=graph_batch[i].edge_mask)
            data_list.append(data)
        batch = Batch.from_data_list(data_list)
        gnn_model = GCN()
        graph_feature = gnn_model(batch)

        ids=batch.batch
        graph_features_list = []
        for i in range(len(graph_batch)):
            graph_features_list.append(graph_feature[ids == i])
        # print(graph_features_list[0].shape)
        x_refine_graph = self.lowChannelOne(x_refine)

        for graph_id, graph_features in enumerate(graph_features_list):
            graph_node_features = graph_batch[graph_id].x
            nodes_scaled = graph_node_features // 4
            # for node in nodes_scaled:
            for node_id, node in enumerate(nodes_scaled):
                nodes_x_64, nodes_y_64 = int(node[0]), int(node[1])
                updated_feature = graph_features[node_id].view(1, 5, 5)

                start_x, end_x = max(nodes_x_64 - 2, 0), min(nodes_x_64 + 3, x_refine_graph.size(2))
                start_y, end_y = max(nodes_y_64 - 2, 0), min(nodes_y_64 + 3, x_refine_graph.size(3))

                if start_x==0:
                    end_x=5
                elif end_x==64:
                    start_x=59

                if start_y==0:
                    end_y=5
                elif end_y==64:
                    start_y=59

                x_refine_graph[graph_id, :, start_x:end_x, start_y:end_y] = updated_feature[:, :, : ]
        x_refine_graph=self.upChannel(x_refine_graph)
        # x_refine64=self.to64Channel(x_refine)
        # x_refine_CA_branch64=self.to64Channel(x_refine_CA_branch)
        # x_refine_topo_branch64=self.to64Channel(x_refine_topo_branch)
        # 所有特征拼接第一次上采样
        # x_cont_refine_in = torch.cat((x_refine64, x_refine_CA_branch64, x_refine_topo_branch64, x_refine_graph), 1)
        x_cont_refine_in = torch.cat((x_refine, x_refine_CA_branch, x_refine_topo_branch, x_refine_graph), 1)
        x_cont_refine = self.refine_up1(x_cont_refine_in)
        x_cont_refine = torch.cat((x_cont_refine, x_refine_down2), 1)
        # 结合之前高维特征第二次上采样
        x_cont_refine = self.refine_up2(x_cont_refine)
        x_cont_refine = torch.cat((x_cont_refine, x_refine_down1), 1)
        # 结合之前高维特征第三次上采样
        x_cont_refine = self.refine_up3(x_cont_refine)
        x_cont_refine = x_cont_refine.tanh()

        return coarse_x_out, x_topo_branch_4vertices_out, x_cont_refine, topo_processed, pixel_mask


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(250, 125)
        self.conv2 = GCNConv(125, 25)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        node_masks, edge_masks = data.node_masks, data.edge_masks
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)  # 使用ReLU作为激活函数
        # x = x * node_masks.unsqueeze(-1)
        return x


class InpaintSANet(torch.nn.Module):
    """
    Inpaint generator, input should be 5*256*256, where 3*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the guidence
    """
    def __init__(self, n_in_channel=11):
        super(InpaintSANet, self).__init__()
        cnum = 32
        self.coarse_net = nn.Sequential(

            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),

            GatedConv2dWithActivation(cnum, 2*cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),

            GatedConv2dWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),

            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum//2, 1, 3, 1, padding=get_pad(128, 3, 1), activation=None)
        )

        self.refine_conv_net = nn.Sequential(

            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),

            GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),

            GatedConv2dWithActivation(2*cnum, 2*cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        )

        self.refine_attn = Self_Attn(4*cnum, 'relu', with_attn=False)

        self.refine_upsample_net = nn.Sequential(
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum//2, 1, 3, 1, padding=get_pad(256, 3, 1), activation=None),
        )
        self.conv_topo = nn.Sequential(
            nn.Conv2d(7, 8, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 1, 3, 1, 1),
        )

    def forward(self, masks=None, ContextRN=None, GuidRefine=None, NDRefine=None,TopoRaw=None,pixel_mask=None,double_output_flag=False,flag_T1=False):
        x_in = torch.cat((ContextRN, GuidRefine, NDRefine, TopoRaw,masks), 1)
        x = self.coarse_net(x_in).tanh()
        coarse_x = x
        if flag_T1:
            return coarse_x

        masked_imgs = ContextRN * (1 - masks) + coarse_x * masks
        x_in_refine = torch.cat((masked_imgs, GuidRefine, NDRefine,TopoRaw, masks), 1)
        x = self.refine_conv_net(x_in_refine)
        x= self.refine_attn(x)
        x = self.refine_upsample_net(x).tanh()
        return  x, coarse_x