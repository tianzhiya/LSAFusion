import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

import Util
from FusionNet import FusionNet
from LowerLightEnhance.LowerLightEnhanceNetwork import LowerLightEnhanceNetwork
from LowerLightEnhance.multi_read_data import MemoryFriendlyLoader
from TaskFusion_dataset import Fusion_dataset
from loss import Fusionloss
from trainEnlightFusion import saveTrainOrTestEnlightenImageY


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def testFuse(type='test'):
    fusion_model_path = './model/Fusion/fusion_model.pth'
    fused_dir = args.fusionResultPath
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
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        start_time = time.time()
        for it, (images_vis, images_ir, name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)

            if args.gpu >= 0:
                images_vis = images_vis.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)
            images_vis_ycrcb = Util.RGB2YCrCb(images_vis)

            num_parameters = count_parameters(fusionmodel)
            num_parameters_m = num_parameters / 1e6  # Convert to M
            print(f"Total parameters: {num_parameters_m:.3f} M")


            import torchprofile
            import warnings

            warnings.filterwarnings("ignore", category=UserWarning)
            macs = torchprofile.profile_macs(fusionmodel, (images_vis_ycrcb,
                                                           images_ir))
            flops = macs * 2  # MACs 转换为 FLOPs
            print(f"FLOPs: {flops / 1e9:.3f} G")  # 转换为 G FLOPs，保留两位小数

            logits, binary_out = fusionmodel(images_vis_ycrcb,
                                             images_ir)
            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :,
                         :], images_vis_ycrcb[:, 2:, :, :]),
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
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('LLIET being processed, please wait .....')

        end_time = time.time()
        avg_time_per_inference = (end_time - start_time) / 361
        fps = 1 / avg_time_per_inference  # FPS in terms of inferences per second
        print(f"Average FPS: {fps:.3f} s")

    pass


def test_Elight():
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
    delete_files_in_directory("./lianHeXuLian/EnlightOut/TestOut")

    with torch.no_grad():
        for _, (input, image_name) in enumerate(enlight_test_queue):
            with torch.no_grad():
                input = Variable(input).cuda()
            inputTrain_ycrcb = Util.RGB2YCrCb(input)
            inputTrainY = inputTrain_ycrcb[:, :1]
            illu_list, ref_list, input_list, atten = model(inputTrainY)
            saveTrainOrTestEnlightenImageY(0, image_name, inputTrain_ycrcb, ref_list)
    pass


import os


def delete_files_in_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    files = os.listdir(directory_path)

    if not files:
        print(f"The directory {directory_path} is already empty.")
        return

    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                print(f"Skipped: {file_path} (not a file)")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--stage', type=int, default=3, help='epochs')
    parser.add_argument('--data_path', type=str, default='./MSRS/Visible/test/MSRS',
                        help='location of the data corpus')
    parser.add_argument('--inlightmodel', type=str, default='./lianHeXuLian/EnlightOut/Model/enhanceVisMode.pt',
                        help='location of the data corpus')
    parser.add_argument('--fusionResultPath', type=str, default='./lianHeXuLian/EnlightOut/MSRSFusion',
                        help='location of the data corpus')
    args = parser.parse_args()
    test_Elight()
    testFuse()
