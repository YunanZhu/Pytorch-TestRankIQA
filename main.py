import os
import argparse
import collections

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision

from model import *
from utils import *


def main(opt):
    print(opt)

    # Set the random seed for reproducing.
    np.random.seed(1)

    # Check GPU state.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("cuda state: " + "available" if use_cuda else "unavailable")

    # Prepare the model.
    net = Vgg16()
    net.load_model(opt.model_file)
    net.to(device)
    net.eval()

    # Load the dataset.
    lines = [line.rstrip('\n') for line in open(opt.test_file)]
    files, mos = [], []
    for i in lines:
        files.append(i.split()[0])
        mos.append(float(i.split()[1]))
    mos = np.asarray(mos)
    print(f"There're {len(files)} images in test set.")

    # Start to test.
    Num_Image = len(files)
    Num_Patch = 30  # Randomly crop patches "Num_Patch" times.
    pred = np.zeros(shape=Num_Image)
    medn = np.zeros(shape=Num_Image)
    with torch.no_grad():
        for i in range(Num_Image):
            # Load image file.
            img = cv2.imread(opt.test_set + files[i])
            img = np.asarray(img)
            x, y = img.shape[0], img.shape[1]

            # Crop patches & calculate pred scores.
            patch_list = []
            for j in range(Num_Patch):
                x_p = np.random.randint(x - 224)
                y_p = np.random.randint(y - 224)
                patch = img[x_p:(x_p + 224), y_p:(y_p + 224), :]
                patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(dim=0).float().to(device)
                patch_list.append(patch)

            patches = torch.cat(patch_list, dim=0)

            # This network can only accept size(224x224) patch.
            score = net(patches)

            pred[i] = torch.mean(score).item()
            medn[i] = torch.median(score).item()

            print(f"{i}: {opt.test_set + files[i]} | {pred[i]} | {medn[i]}")

    PLCC = plcc(pred, mos)
    SROCC = srocc(pred, mos)
    print(f"PLCC = {PLCC:.4f}, SROCC = {SROCC:.4f}")

    if (opt.res_file is not None) and (opt.res_file.lower() != "none"):
        with open(opt.res_file, mode='w') as f:
            print(f"img_file,mos,pred,plcc,{PLCC:.4f},srcc,{SROCC:.4f}", file=f)
            for (im, gt, pd) in zip(files, mos, pred):
                print(f"{im},{gt},{pd}", file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # "/your_own_datasets_path/TID2013/"
    # "/your_own_datasets_path/LIVE2/"
    parser.add_argument("--test_set", type=str, default=None, help="test set path")

    # "./pre-trained/Rank_tid2013.caffemodel.pt"
    # "./pre-trained/FT_tid2013.caffemodel.pt"
    # "./pre-trained/Rank_live.caffemodel.pt"
    # "./pre-trained/FT_live.caffemodel.pt"
    parser.add_argument("--model_file", type=str, default=None, help="trained model file")

    # "./data/ft_tid2013_test.txt"
    # "./data/ft_live_test.txt"
    parser.add_argument("--test_file", type=str, default=None, help="file to store MOS and image filenames")

    # "./result.csv"
    parser.add_argument("--res_file", type=str, default=None, help="csv file to save the pred scores")

    main(parser.parse_args())
