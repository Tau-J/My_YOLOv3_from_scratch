from __future__ import division
import time
import torch 
import torch.nn as nn
import numpy as np
import cv2 
from utils import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='imgs', type=str)
    parser.add_argument('--det', default='det', type=str)
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--confidence', default=0.5, type=float)
    parser.add_argument('--nms_thresh', default=0.4, type=float)
    parser.add_argument('--cfg', default='cfg/yolov3.cfg', type=str)
    parser.add_argument('--weights', default='', type=str)
    parser.add_argument('--resolution', default=416, type=int)

    return parser.parse_args()

args = arg_parse()
images = args.images
det = args.det
bs = args.bs
device = torch.device('cuda:' + args.gpu)
confidence = args.confidence
nms_thresh = args.nms_thresh
cfg = args.cfg
weights = args.weights
resolution = args.resolution

start = 0
use_gpu = torch.cuda.is_available()
classes = load_classes('data/coco.names')
num_classes = len(classes) # coco = 80

print('loading model...')
model = Darknet(cfg)
model.load_weights(weights)
print('model loaded.')

model.net_info['height'] = resolution
input_size = resolution
assert input_size % 32 == 0 and input_size > 32

if use_gpu:
    model = model.to(device)

model.eval()

read_dir = time.time()
# Detection phase
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()

if not osp.exists(det):
    os.makedirs(det)

load_batch = time.time()
loaded_imgs = [cv2.imread(i) for i in imlist]

img_batches = list(map(prep_image, loaded_imgs, [size] * len(imlist)]))

# List containing dimensions of original images
img_size_list = [(x.shape[1], x.shape[0]) for x in loaded_imgs]
img_size_list = torch.FloatTensor(img_size_list).repeat(1,2)

if use_gpu:
    img_size_list = img_size_list.to(device)

leftover = int(len(imlist) % bs != 0)
if bs != 1:
    num_batches = len(imlist) // bs + leftover
    img_batches = [torch.cat((img_batches[i*bs:min((i+1)*bs, len(img_batches))]))
        for x in range(num_batches)]
