import cv2
import numpy as np
import argparse
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default='./MSRS/Visible/train/MSRS')
parser.add_argument('--save_path', type=str, default='./MSRS/Visible/train/VisDark')
args = parser.parse_args()

img_path = sorted(glob(args.load_path + '/*'))
img_name = sorted(os.listdir(args.load_path))


def simulateLowLightImages(img, add_noise=True, mean=0, std=0.03, reduce_brightness=True, gamma=2.2):
    if add_noise:
        img = img + np.random.normal(mean, std, img.shape)
    if reduce_brightness:
        img = np.power(img, gamma)
    return img


for img_path, img_name in zip(img_path, img_name):
    img = cv2.imread(img_path) / 255.
    img = simulateLowLightImages(img, add_noise=False, gamma=6)
    cv2.imwrite(args.save_path + '/' + img_name, img * 255)
