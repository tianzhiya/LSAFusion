import torch
import torch.nn as nn

import Util

import torch.nn.functional as F
import numpy as np

from LowerLightEnhance.loss import LightLossFunction, ColorAngleLossModule


class LowerLightEnhanceNetwork(nn.Module):

    def __init__(self, stage=3):
        super(LowerLightEnhanceNetwork, self).__init__()
        self.stage = stage
        self.enhance = EnhanceNetwork(layers=1, channels=1)
        self.calibrate = CalibrateNetwork(layers=3, channels=16)
        self._criterion = LightLossFunction()
        self.colorAngleLoss = ColorAngleLossModule()
        self.grm = GRMModule(1, 1)
        sobelxyEnhanceInchannels = 3
        self.sobelconv = Sobelxy(sobelxyEnhanceInchannels)
        self.sobelconvup = Conv1(3, 3)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, OriginalImage):

        enhancelist, rlist, perOPInputlist, attlist = [], [], [], []
        input_op = OriginalImage
        for i in range(self.stage):
            perOPInputlist.append(input_op)
            i = self.enhance(input_op)
            # i = self.grm(i)
            r = OriginalImage / i
            r = torch.clamp(r, 0, 1)

            att = self.calibrate(r)
            input_op = OriginalImage + att
            enhancelist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        return enhancelist, rlist, perOPInputlist, attlist

    def _loss(self, input, inputTrain_ycrcb):
        enhancelist, en_list, perOPInputlist, _ = self(input)
        loss = 0
        for i in range(self.stage):
            loss += self._criterion(perOPInputlist[i], enhancelist[i])
        inputTrain_RGB = Util.YCrCb2RGB(inputTrain_ycrcb)
        fuseImageRgb = self.get_if_tensor(en_list[0], inputTrain_RGB)
        colorLoss = self.colorAngleLoss(fuseImageRgb, inputTrain_RGB)
        return loss + colorLoss

    def get_if_tensor(self, Yf, vi_3):
        # Convert vi_3 from RGB to YCbCr
        vi_ycbcr = Util.RGB2YCrCb(vi_3)

        # # Extract Cb and Cr channels and expand dimensions
        # cb = vi_ycbcr[:, :, :, 1].unsqueeze(-1)
        # cr = vi_ycbcr[:, :, :, 2].unsqueeze(-1)
        #
        # # Concatenate Yf, Cb, and Cr to form If in YCbCr space
        # If_ycbcr = torch.cat([Yf, cb, cr], dim=-1)

        If_ycbcr = torch.cat(
            (Yf, vi_ycbcr[:, 1:2, :,
                 :], vi_ycbcr[:, 2:, :, :]),
            dim=1,
        )

        # Convert If back to RGB
        If_rgb = Util.YCrCb2RGB(If_ycbcr)

        return If_rgb


class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
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
        sobelxyEnhanceInchannels = 1
        self.sobelconv = Sobelxy(sobelxyEnhanceInchannels)
        self.sobelconvup = Conv1(1, 1)

    def forward(self, input):
        sobel = self.sobelconv(input)
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input

        sobelconvup = self.sobelconvup(sobel)
        illu = sobelconvup + illu
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
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
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


class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self._criterion = LightLossFunction()

        base_weights = torch.load(weights)
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        i = self.enhance(input)
        r = input / i
        r = torch.clamp(r, 0, 1)
        return i, r

    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, i)
        return loss


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
        final_output = torch.clamp(final_output, 0, 1)
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
