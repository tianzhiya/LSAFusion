# This is a sample Python script.
import argparse
import datetime
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

import Util
from FusionNet import FusionNet
from TaskFusion_dataset import Fusion_dataset
from loss import Fusionloss


def train_fusion(i):
    lr_start = 0.001
    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    fusionmodel = FusionNet(output=1)
    fusionmodel.cuda()
    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    train_dataset = Fusion_dataset('train')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    print(train_loader.n_iter)
    criteria_fusion = Fusionloss()

    epoch = 10
    benCiStartTime = glob_StartTime = time.time()
    print("Train Fusion Model start...")

    for epo in range(0, epoch):
        for it, (image_vis, image_ir, name, bi) in enumerate(train_loader):
            fusionmodel.train()
            # image_vis_dark = simulateLowLightImages(image_vis, add_noise=False, gamma=6)
            # image_vis_dark = Variable(image_vis_dark).cuda()
            # image_vis_dark_ycrcb = RGB2YCrCb(image_vis_dark)
            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = Util.RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()
            bi = Variable(bi).cuda()

            generateImageY, biPredict = fusionmodel(
                image_vis,
                image_ir)
            optimizer.zero_grad()
            # fusion loss
            loss_fusion, loss_in, loss_grad, bi_loss = criteria_fusion(
                image_vis_ycrcb, image_ir, generateImageY, biPredict,
                bi
            )

            loss_total = loss_fusion
            loss_total.backward()
            optimizer.step()
            endTime = time.time()
            t_intv, glob_t_intv = endTime - benCiStartTime, endTime - glob_StartTime
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'bi_loss: {bi_loss:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    bi_loss=bi_loss.item(),
                    time=t_intv,
                    eta=eta,
                )
                print(msg)
                benCiStartTime = endTime

    fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
    torch.save(fusionmodel.state_dict(), fusion_model_file)


def simulateLowLightImages(img, add_noise=True, mean=0, std=0.03, reduce_brightness=True, gamma=2.2):
    if add_noise:
        img = img + np.random.normal(mean, std, img.shape)
    if reduce_brightness:
        img = np.power(img, gamma)
    return img


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def run_fusion(type='test'):
    fusion_model_path = './model/Fusion/fusion_model.pth'
    fused_dir = os.path.join('./MSRS/Fusion', type, 'MSRS')
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusionmodel = eval('FusionNet')(output=1)
    fusionmodel.eval()
    # if args.gpu >= 0:
    # fusionmodel.cuda(args.gpu)
    fusionmodel.cuda()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('done!')
    test_dataset = Fusion_dataset(type)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)

            if args.gpu >= 0:
                images_vis = images_vis.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)

            images_vis_ycrcb = Util.RGB2YCrCb(images_vis)
            logits, binary_out = fusionmodel(images_vis_ycrcb,
                                             images_ir)
            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :,
                         :], images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='EnlightFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    print(args.gpu)
    print(args.model_name)
    print(args.batch_size)

    for i in range(4):
        train_fusion(i)
    with torch.no_grad():
        torch.cuda.empty_cache()
        run_fusion()
