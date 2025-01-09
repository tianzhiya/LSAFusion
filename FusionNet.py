# coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RemarkObjectGuide import SIM
from ResizeToSquare import ResizeToSquare
from Transformer.swin_transformer import SwinTransformer


class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5


class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ReduceSwinTransformerC(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(ReduceSwinTransformerC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2 * channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x


class RGBD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RGBD, self).__init__()
        self.dense = DenseBlock(in_channels)
        self.convdown = Conv1(3 * in_channels, out_channels)
        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels, out_channels)

    def forward(self, x):
        x1 = self.dense(x)
        x1 = self.convdown(x1)
        x2 = self.sobelconv(x)
        x2 = self.convup(x2)
        return F.leaky_relu(x1 + x2, negative_slope=0.1)


class FusionNet(nn.Module):
    def __init__(self, output):
        super(FusionNet, self).__init__()
        vis_ch = [16, 32, 48]
        inf_ch = [16, 32, 48]
        output = 1
        self.vis_conv = ConvLeakyRelu2d(1, vis_ch[0])
        self.vis_rgbd1 = RGBD(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = RGBD(vis_ch[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv = ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = RGBD(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = RGBD(inf_ch[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2] + inf_ch[2] + inf_ch[2], vis_ch[1] + vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1] + inf_ch[1], vis_ch[0] + inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0] + inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)

        self.enhance = EnhanceNetwork(layers=1, channels=32)
        self.calibrate = CalibrateNetwork(layers=3, channels=16)
        self.visCoarseConv = ConvBnTanh2d(vis_ch[0], 1)
        self.stage = 3
        self.grm = GRMModule(vis_ch[0], vis_ch[0])
        self.swinTransformerOutPutChanel = 192 + 192
        self.s2PM = S2PM(self.swinTransformerOutPutChanel, inf_ch[2])
        self.s2p2 = S2P2(inf_ch[2])
        self.sim1 = SIM(norm_nc=64, label_nc=48, nhidden=32)
        self.sim2 = SIM(norm_nc=32, label_nc=48, nhidden=32)
        self.resizeToSquare = ResizeToSquare(target_size=256)
        self.swinTransformer = SwinTransformer(
            hidden_dim=96,
            layers=(2, 2, 6, 2),
            heads=(3, 6, 12, 24),
            channels=1,
            num_classes=3,
            head_dim=32,
            window_size=1,
            downscaling_factors=(4, 2, 2, 2),
            relative_pos_embedding=True
        )

        self.reduceSwinTransformerC = ReduceSwinTransformerC(self.swinTransformerOutPutChanel, inf_ch[2])

        self.visTSFReshape1 = UpsampleAndAdjustChannels(in_channels=96, out_channels=32)
        self.visTSFReshape2 = UpsampleAndAdjustChannels(in_channels=192, out_channels=48)
        self.visSEM1 = SEM(in_dim=32, out_dim=32)
        self.visSEM2 = SEM(in_dim=48, out_dim=48)

        self.irTSFReshape1 = UpsampleAndAdjustChannels(in_channels=96, out_channels=32)
        self.irTSFReshape2 = UpsampleAndAdjustChannels(in_channels=192, out_channels=48)
        self.irSEM1 = SEM(in_dim=32, out_dim=32)
        self.irSEM2 = SEM(in_dim=48, out_dim=48)

    def forward(self, image_vis, image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:, :1]
        x_inf_origin = image_ir

        height, width = image_vis.size(2), image_vis.size(3)

        x_vis_p = self.vis_conv(x_vis_origin)
        # ----------------------------------------------------------------
        # enhancelist, enhanceNoromlist, inlist, attlist = [], [], [], []
        # input = x_vis_p
        # input_op = input
        # for i in range(self.stage):
        #     inlist.append(self.visCoarseConv(input_op))
        #     i = self.enhance(input_op)
        #     r = input / i
        #     r = torch.clamp(r, 0, 1)
        #     r = self.grm(r)
        #     r = torch.clamp(r, 0, 1)
        #     att = self.calibrate(r)
        #     input_op = input + att
        #     enhancelist.append(self.visCoarseConv(i))
        #     enhanceNoromlist.append(self.visCoarseConv(r))
        #     attlist.append(torch.abs(self.visCoarseConv(att)))

        visTsfDownsampled_1, visTsfDownsampled_2 = self.swinTransformer(x_vis_origin)
        target_size = (height, width)
        visTSFReshape1 = self.visTSFReshape1(visTsfDownsampled_1, target_size)
        visTSFReshape2 = self.visTSFReshape2(visTsfDownsampled_2, target_size)
        visTsfUpsampled_1 = F.interpolate(visTsfDownsampled_2, size=(height, width), mode='bilinear',
                                          align_corners=False)

        infTsfDownsampled_1, infTsfDownsampled_2 = self.swinTransformer(x_inf_origin)
        infTSFReshape1 = self.irTSFReshape1(infTsfDownsampled_1, target_size)
        infTSFReshape2 = self.irTSFReshape2(infTsfDownsampled_2, target_size)
        infTsfUpsampled_1 = F.interpolate(infTsfDownsampled_2, size=(height, width), mode='bilinear',
                                          align_corners=False)

        # encode
        x_vis_p1 = self.vis_rgbd1(x_vis_p)
        x_vis_p1 = self.visSEM1(x_vis_p1, visTSFReshape1)
        x_vis_p2 = self.vis_rgbd2(x_vis_p1)
        x_vis_p2 = self.visSEM2(x_vis_p2, visTSFReshape2)

        x_inf_p = self.inf_conv(x_inf_origin)

        x_inf_p1 = self.inf_rgbd1(x_inf_p)
        x_inf_p1 = self.irSEM1(x_inf_p1, infTSFReshape1)
        x_inf_p2 = self.inf_rgbd2(x_inf_p1)
        x_inf_p2 = self.irSEM2(x_inf_p2, infTSFReshape2)

        irVisUpsampledCat = torch.cat((infTsfUpsampled_1, visTsfUpsampled_1), dim=1)
        seg_f = self.s2PM(irVisUpsampledCat)
        binary_out = self.s2p2(seg_f)
        reduceSwinTransformerC = self.reduceSwinTransformerC(irVisUpsampledCat)

        catF = torch.cat((x_vis_p2, x_inf_p2, reduceSwinTransformerC), dim=1)

        # decode
        x = self.decode4(catF)
        features_seg_rec1 = self.sim1(x, seg_f)
        x = self.decode3(features_seg_rec1)
        features_seg_rec2 = self.sim2(x, seg_f)
        x = self.decode2(features_seg_rec2)
        x = self.decode1(x)
        return x, binary_out


class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)
        return illu


class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        delta = input - fea

        return delta


class GRMModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GRMModule, self).__init__()
        # 主要流: 两个3x3的卷积层，带有LeakyReLU，以及一个1x1的卷积层。
        self.main_stream = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        # 第一个残差流: Sobel算子和1x1卷积层。
        self.sobel_operator = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 第二个残差流: Laplacian算子。
        self.laplacian_operator = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 主要流
        main_out = self.main_stream(x)
        # 第一个残差流
        sobel_out = self.gradient_feature(x)
        residual_out = self.residual_conv(x)
        # 第二个残差流
        laplacian_out = self.laplacian(x)
        # 纹理增强的第一阶段
        enhanced_texture = main_out + laplacian_out
        # 合并主要流和第一个残差流
        combined_out = main_out + sobel_out + residual_out
        # 纹理增强的第二阶段
        final_output = enhanced_texture + combined_out
        return final_output

    def gradient_feature(self, input_tensor):
        # 计算Sobel卷积核的大小，每个通道的大小都相同
        sobel_kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_kernel_x = sobel_kernel_x.view(1, 1, 3, 3).expand(-1, input_tensor.size(1), -1, -1).to('cuda')
        sobel_kernel_y = sobel_kernel_x.permute(0, 1, 3, 2)
        gradient_orig_x = torch.abs(F.conv2d(input_tensor, sobel_kernel_x, padding=1))
        gradient_orig_y = torch.abs(F.conv2d(input_tensor, sobel_kernel_y, padding=1))
        grad_min_x = torch.min(gradient_orig_x)
        grad_max_x = torch.max(gradient_orig_x)
        grad_min_y = torch.min(gradient_orig_y)
        grad_max_y = torch.max(gradient_orig_y)
        grad_norm_x = (gradient_orig_x - grad_min_x) / (grad_max_x - grad_min_x + 0.0001)
        grad_norm_y = (gradient_orig_y - grad_min_y) / (grad_max_y - grad_min_y + 0.0001)
        grad_norm = grad_norm_x + grad_norm_y
        return grad_norm

    def laplacian(self, input_tensor):
        # 计算Laplacian卷积核的大小，每个通道的大小都相同
        laplacian_kernel = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).expand(-1, input_tensor.size(1), -1, -1).to('cuda')
        gradient_orig = torch.abs(F.conv2d(input_tensor, laplacian_kernel, padding=1))
        grad_min = torch.min(gradient_orig)
        grad_max = torch.max(gradient_orig)
        grad_norm = (gradient_orig - grad_min) / (grad_max - grad_min + 0.0001)
        return grad_norm


class S2PM(nn.Module):
    def __init__(self, in_channel=48, out_channel=48):
        super(S2PM, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        out = self.block3(x2)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class S2P2(nn.Module):
    '''This path plays the role of a classifier and is responsible for predicting the results of semantic segmentation, binary segmentation and edge segmentation'''

    def __init__(self, feature=64):
        super(S2P2, self).__init__()
        self.binary_conv1 = ConvBNReLU(feature, feature // 4, kernel_size=1)
        self.binary_conv2 = ConvBNReLU(feature // 4, feature // 6, kernel_size=1)
        self.binary_conv3 = nn.Conv2d(feature // 6, 2, kernel_size=3, padding=1)

        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, feat):
        binary = self.binary_conv3(self.binary_conv2(self.binary_conv1(feat)))
        return binary


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1, bn=True, relu=True):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                              bias=False if bn else True)
        self.bn = bn
        if bn:
            self.bnop = nn.BatchNorm2d(out_planes)
        self.relu = relu
        if relu:
            self.reluop = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bnop(x)
        if self.relu:
            x = self.reluop(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# 基于特征刷选的融合网络
def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# ========= Saliency-enhanced Module ========= #
class SEM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SEM, self).__init__()

        act_fn = nn.ReLU(inplace=True)

        self.layer_rgb = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), act_fn)
        self.layer_t = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), act_fn)

        self.gamma_rf = nn.Parameter(torch.zeros(1))
        self.gamma_tf = nn.Parameter(torch.zeros(1))

        # self.gamma_rf = 1.0
        # self.gamma_tf = 1.0

        self.layer_transF = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, transF):
        x_rgb = self.layer_rgb(rgb)
        x_transF = self.layer_transF(transF)
        att_transF = nn.Sigmoid()(x_transF)
        x_rgb_W = x_rgb.mul(att_transF)
        x_rgb_en = self.gamma_rf * x_rgb_W + rgb
        return x_rgb_en


class UpsampleAndAdjustChannels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleAndAdjustChannels, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, input_tensor, target_size):
        # 上采样到目标大小
        upsampled_tensor = F.interpolate(input_tensor, size=target_size, mode='bilinear', align_corners=False)

        # 使用 1x1 卷积调整通道数
        output_tensor = self.conv1x1(upsampled_tensor)
        return output_tensor


class FuseModule(nn.Module):
    """ Interactive fusion module"""

    def __init__(self, in_dim=64):
        super(FuseModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, prior):
        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        attention = self.sig(energy)
        attention_x = x * attention
        attention_p = prior * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        p_gamma = self.gamma2(torch.cat((prior, attention_p), dim=1))
        prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

        return x_out, prior_out
