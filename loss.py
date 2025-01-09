import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Util
from math import exp
from torch.autograd import Variable


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class NormalLoss(nn.Module):
    def __init__(self, ignore_lb=255, *args, **kwargs):
        super(NormalLoss, self).__init__()
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        return torch.mean(loss)


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generateImageY, bi_pred, bi):
        image_y = image_vis[:, :1, :, :]

        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generateImageY)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generateImageY)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        contentLoss = loss_in + 10 * loss_grad

        bi_loss = self.binarySegLoss(bi, bi_pred)
        loss_total = contentLoss + bi_loss
        return loss_total, loss_in, loss_grad, bi_loss

    def binarySegLoss(self, bi, bi_pred):
        bi = bi.to('cpu')  # 或者 .to('cpu')，取决于您的需求
        bi_pred = bi_pred.to('cpu')  # 或者 .to('cpu')
        # 将标签 `bi` 转换为 one-hot 编码
        # 将类别值的张量转换为索引张量
        bi_indices = bi.long()  # 将张量中的值转换为整数类型

        bi = F.one_hot(bi_indices, num_classes=2)
        # 将 one-hot 编码的 `bi` 调整维度以匹配 `bi_pred`
        bi = bi.permute(0, 3, 1, 2).float()
        # 定义二元交叉熵损失的类别权重
        binary_class_weight = np.array([1.4548, 19.8962])
        binary_class_weight = torch.tensor(binary_class_weight).float().to('cpu')
        binary_class_weight = binary_class_weight.unsqueeze(0)
        binary_class_weight = binary_class_weight.unsqueeze(2)
        binary_class_weight = binary_class_weight.unsqueeze(2)
        # 使用带权重的二元交叉熵损失函数
        bi_loss = F.binary_cross_entropy_with_logits(bi_pred, bi, pos_weight=binary_class_weight)
        return bi_loss

    def fromYcrcbImage(self, generateImageY, image_vis):
        fromYcrcbImage = torch.cat(
            (generateImageY, image_vis[:, 1:2, :,
                             :], image_vis[:, 2:, :, :]),
            dim=1,
        )

        return fromYcrcbImage


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


if __name__ == '__main__':
    pass
