import os
import sys
import time
import glob
import numpy as np
import torch

import Util

from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

from LowerLightEnhance import utils
from LowerLightEnhance.LowerLightEnhanceNetwork import LowerLightEnhanceNetwork
from LowerLightEnhance.multi_read_data import MemoryFriendlyLoader
from model import *

parser = argparse.ArgumentParser("LLIET")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=1000, help='epochs')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--stage', type=int, default=3, help='epochs')
parser.add_argument('--save', type=str, default='Result/', help='location of the data corpus')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
image_path = args.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)
mModelSavePath = '../LowerLightEnhance/EnlightOut/Model/'
mTestOutSavePath = '../LowerLightEnhance/EnlightOut/TestOut/'
mTrainOutSavePath = '../LowerLightEnhance/EnlightOut/TrainOut/'

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    model = LowerLightEnhanceNetwork(stage=args.stage)

    model.enhance.in_conv.apply(model.weights_init)
    model.enhance.conv.apply(model.weights_init)
    model.enhance.out_conv.apply(model.weights_init)
    model.calibrate.in_conv.apply(model.weights_init)
    model.calibrate.convs.apply(model.weights_init)
    model.calibrate.out_conv.apply(model.weights_init)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    print(MB)
    train_low_data_names = '../MSRS/Visible/test/MSRS'
    # train_low_data_names = '../MSRS/Visible/test/test1'
    TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')

    test_low_data_names = '../MSRS/Visible/test/MSRS'
    # test_low_data_names = '../MSRS/Visible/test/test1'

    TestDataset = MemoryFriendlyLoader(img_dir=test_low_data_names, task='test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = torch.Generator(device='cuda')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=True, generator=generator)

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=True, generator=generator)

    total_step = 0

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for batch_idx, (input, _) in enumerate(train_queue):
            total_step += 1
            inputTrain = Variable(input, requires_grad=False).cuda()
            inputTrain_ycrcb = Util.RGB2YCrCb(inputTrain)
            inputTrainY = inputTrain_ycrcb[:, :1]

            optimizer.zero_grad()
            loss = model._loss(inputTrainY, inputTrain_ycrcb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            losses.append(loss.item())
            print('train-epoch {:03d} {:03d} {:.6f}'.format(epoch, batch_idx, loss))

        logging.info('train-epoch %03d %f', epoch, np.average(losses))

        utils.save(model, os.path.join(mModelSavePath, 'enhanceVisMode.pt'))
        if epoch % 1 == 0 and epoch % 3 == 0:
            logging.info('train %03d %f', epoch, loss)
            model.eval()
            with torch.no_grad():
                for _, (input, image_name) in enumerate(test_queue):
                    # input = Variable(input, volatile=True).cuda()
                    with torch.no_grad():
                        inputTest = input.cuda()
                        inputTest_ycrcb = Util.RGB2YCrCb(inputTest)
                        inputTestY = inputTest_ycrcb[:, :1]

                    illu_list, ref_list, input_list, atten = model(inputTestY)
                    if epoch == 1 or epoch % 3 == 0:
                        saveEnlightenImageY(epoch, image_name, inputTest_ycrcb, ref_list)


def saveEnlightenImageY(epoch, image_name, inputTest_ycrcb, ref_list):
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
    for k in range(len(image_name)):
        image = fused_image[k, :, :, :]
        image = image.squeeze()
        image = Image.fromarray(image)

        u_name = '%s' % (image_name[k])
        u_path = image_path + '/' + u_name
        image.save(u_path)
        print('LLIET {0} Sucessfully!'.format(u_path))


if __name__ == '__main__':
    main()
