import sys
import logging

import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn

from LowerLightEnhance import utils
from LowerLightEnhance.LowerLightEnhanceNetwork import LowerLightEnhanceNetwork, Finetunemodel
from LowerLightEnhance.multi_read_data import MemoryFriendlyLoader

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

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=int, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=3, help='epochs')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--stage', type=int, default=3, help='epochs')
parser.add_argument('--data_path', type=str, default='./MSRS/Visible/test/MSRS',
                    help='location of the data corpus')
parser.add_argument('--inlightmodel', type=str, default='./lianHeXuLian/EnlightOut/Model/enhanceVisMode.pt',
                    help='location of the data corpus')

args = parser.parse_args()
mModelSavePath = './lianHeXuLian/EnlightOut/Model/'
mTestOutSavePath = './lianHeXuLian/EnlightOut/TestOut/'
mTrainOutSavePath = './lianHeXuLian/EnlightOut/TrainOut/'

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def setGPU():
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                  "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')


def train_Elight(num=0):
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    model = LowerLightEnhanceNetwork(stage=args.stage)
    model.enhance.in_conv.apply(model.weights_init)
    model.enhance.conv.apply(model.weights_init)
    model.enhance.out_conv.apply(model.weights_init)
    model.calibrate.in_conv.apply(model.weights_init)
    model.calibrate.convs.apply(model.weights_init)
    model.calibrate.out_conv.apply(model.weights_init)
    model = model.cuda()

    total_step = 0
    #######train
    lr_start = 0.001
    fusionmodel = FusionNet(output=1)
    fusionmodel.cuda()
    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    fusion_train_dataset = Fusion_dataset('train')
    fusion_train_loader = DataLoader(
        dataset=fusion_train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
    )
    if num > 0:
        save_pth = './model/Fusion/fusion_model.pth'
        # todo: define model and frozen parameters
        fusionmodel = FusionNet(output=1)
        fusionmodel.cuda()
        fusionmodel.load_state_dict(torch.load(save_pth))
        fusionmodel.eval()
        for p in fusionmodel.parameters():
            p.requires_grad = False
        logging.info('Load SOD Model {} Sucessfully~'.format(save_pth))
        # todo: define loss functions for SOD
        fusion_train_loader.n_iter = len(fusion_train_loader)
        print(fusion_train_loader.n_iter)

    criteria_fusion = Fusionloss()
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for it, (image_vis, image_ir, name, bi) in enumerate(fusion_train_loader):
            total_step += 1
            inputTrain = Variable(image_vis, requires_grad=False).cuda()
            inputTrain_ycrcb = Util.RGB2YCrCb(inputTrain)
            image_ir = Variable(image_ir).cuda()
            inputTrainY = inputTrain_ycrcb[:, :1]

            optimizer.zero_grad()
            loss = model._loss(inputTrainY, inputTrain_ycrcb)

            generateImageY, biPredict = fusionmodel(
                inputTrain_ycrcb,
                image_ir)
            optimizer.zero_grad()
            # fusion loss
            loss_fusion, loss_in, loss_grad, bi_loss = criteria_fusion(
                inputTrain_ycrcb, image_ir, generateImageY, biPredict,
                bi
            )

            rongHeloss_total = loss_fusion

            if num > 0:
                lossTotal = loss + rongHeloss_total
            else:
                lossTotal = loss

            lossTotal.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
        utils.save(model, os.path.join(mModelSavePath, 'enhanceVisMode.pt'))

        if epoch % 1 == 0 and epoch % 3 == 0:
            logging.info('train %03d %f', epoch, loss)
            model.eval()
            with torch.no_grad():
                for it, (image_vis, image_ir, name, bi) in enumerate(fusion_train_loader):
                    # input = Variable(input, volatile=True).cuda()
                    with torch.no_grad():
                        inputTest = image_vis.cuda()
                        inputTest_ycrcb = Util.RGB2YCrCb(inputTest)
                        inputTestY = inputTest_ycrcb[:, :1]
                    illu_list, ref_list, input_list, atten = model(inputTestY)
                    if epoch == 1 or epoch % 3 == 0:
                        saveTrainOrTestEnlightenImageY(1, name, inputTest_ycrcb, ref_list)
    pass


def saveTrainOrTestEnlightenImageY(isTrainIsTest, image_name, inputTest_ycrcb, ref_list):
    fusion_ycrcb = torch.cat(
        (ref_list[0], inputTest_ycrcb[:, 1:2, :,
                      :], inputTest_ycrcb[:, 2:, :, :]),
        dim=1,
    )
    fusion_image = Util.YCrCb2RGB(fusion_ycrcb)
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
    fused_dir = os.path.join('./LowerLightEnhance/', 'out')
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    for k in range(len(image_name)):
        image = fused_image[k, :, :, :]
        image = image.squeeze()
        image = Image.fromarray(image)
        u_name = '%s' % (image_name[k])
        if isTrainIsTest:
            u_path = mTrainOutSavePath + '/' + u_name
            if not os.path.exists(mTrainOutSavePath):
                os.makedirs(mTrainOutSavePath)
        else:
            u_path = mTestOutSavePath + '/' + u_name
            if not os.path.exists(mTestOutSavePath):
                os.makedirs(mTestOutSavePath)

        image.save(u_path)
        print('LLIET being processed, please wait .....')


def run_Elight():
    model = LowerLightEnhanceNetwork(stage=args.stage)
    model.eval()
    model.cuda()
    model.load_state_dict(torch.load(args.inlightmodel))
    print('done!')

    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')
    enlight_test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=False, num_workers=0)

    model.eval()
    with torch.no_grad():
        for _, (input, image_name) in enumerate(enlight_test_queue):
            with torch.no_grad():
                input = Variable(input).cuda()
            inputTrain_ycrcb = Util.RGB2YCrCb(input)
            inputTrainY = inputTrain_ycrcb[:, :1]
            illu_list, ref_list, input_list, atten = model(inputTrainY)
            saveTrainOrTestEnlightenImageY(0, image_name, inputTrain_ycrcb, ref_list)
    pass


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def train_Fuse(i):
    for i in range(4):
        train_fusion(i)
    with torch.no_grad():
        torch.cuda.empty_cache()
        run_fusion()
    pass


def train_fusion(i):
    lr_start = 0.001
    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    fusionmodel = FusionNet(output=1)

    fusionmodel.cuda()
    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    train_dataset = Fusion_dataset('enLightIntrain')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
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
            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = Util.RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()
            bi = Variable(bi).cuda()

            generateImageY, biPredict = fusionmodel(
                image_vis,
                image_ir)
            optimizer.zero_grad()
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
    fusionmodel.cuda()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('done!')
    test_dataset = Fusion_dataset(type)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
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


def test_Fuse():
    type = 'test'
    fusion_model_path = './model/Fusion/fusion_model.pth'
    fused_dir = os.path.join('./lianHeXuLian/EnlightOut', 'Fusion')
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusionmodel = eval('FusionNet')(output=1)
    fusionmodel.eval()

    fusionmodel.cuda()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))

    test_dataset = Fusion_dataset(type)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
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


if __name__ == "__main__":
    alternate = 1
    for i in range(0, alternate):
        train_Elight(i)
        print("|{0} LLIET being trained~!".format(i + 1))
        run_Elight()
        train_Fuse(i)
    print("training Done!")

    test_Fuse()
