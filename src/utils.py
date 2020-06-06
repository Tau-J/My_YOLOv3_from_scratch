from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import cv2 

def transform_predict(pred, img_size, anchors, num_classes, device=None):
    '''
    Takes the prediction featuremap and some params

    Return a 2-dim tensor (BHW)x(5+C) which reshape from the prediction
    C = num_classes
    B = len(anchors)
    H = W = pred.size(2) = pred.size(3)
    '''
    batch_size = pred.size(0)
    scale = img_size // pred.size(2) # original img size is 'scale' times largger than the pred size(due to conv)
    grid_size = pred.size(2)         # current grid size is cur_size*cur_size
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # we want to reshape bs, (5+C)*B, H, W -> bs, BHW, (5+C)
    # step1: bs, (5+C)*B, H, W -> bs, (5+C)*B, HW 
    pred = pred.reshape(batch_size, bbox_attrs, grid_size*grid_size*num_anchors)
    # step2: bs, (5+C)*B, HW -> bs, HW, (5+C)*B
    pred = pred.transpose(1, 2)
    # step3: bs, HW, (5+C)*B -> bs, BHW, (5+C)
    pred = pred.reshape(batch_size, num_anchors*grid_size*grid_size, bbox_attrs)

    anchors = [(a[0]/scale, a[1]/scale) for a in anchors]

    # 5+C = tx, ty, tw, th, ob, c1, c2 ... cn
    pred[:,:0] = pred[:,:0].sigmoid() # tx
    pred[:,:1] = pred[:,:1].sigmoid() # ty
    pred[:,:4] = pred[:,:4].sigmoid() # ob

    grid = np.arange(grid_size)
    x, y = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(x).reshape(-1,1) # g x 1
    y_offset = torch.FloatTensor(y).reshape(-1,1) # g x 1

    if device: # use GPU
        x_offset = x_offset.to(device)
        y_offset = y_offset.to(device)
    
    # concat -> g, 2
    x_y_offset = torch.cat((x_offset, y_offset), 1) 
    # g, 2 -> g, 2xB
    x_y_offset = x_y_offset.repeat(1, num_anchors)
    # g, 2xB -> gxB, 2
    x_y_offset = x_y_offset.reshape(-1, 2)
    # gxB, 2 -> 1, gxB, 2
    x_y_offset = x_y_offset.unsqueeze(0)

    pred[:,:,:2] += x_y_offset # add offset to tx, ty

    anchors = torch.FloatTensor(anchors)
    if device:
        anchors = anchors.to(device)
    
    # init anchors for every grid unit
    # 1 x B -> HW, B
    anchors = anchors.repeat(grid_size*grid_size, 1)
    # HW, B -> 1, HW, B
    anchors = anchors.unsqueeze(0)
    # th = exp(th) * anchor_th (tw too)
    pred[:,:,2:4] = torch.exp(pred[:,:,2:4])*anchors

    # p(c) = sigmoid(c)
    pred[:,:,5:5+num_classes] = pred[:,:,5:5+num_classes].sigmoid()

    pred[:,:,:4] *= scale # bbox in origial img

    return pred