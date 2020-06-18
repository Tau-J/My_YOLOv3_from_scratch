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

def load_classes(names_file):
    with open(names_file, 'r') as f:
        names = f.read.split('\n')[:-1]
    return names

def nms(bboxes, conf_score, thresh):
    res_bboxes_idx = []
    if len(bboxes):
        bbox_x1 = bboxes[:,0]
        bbox_x2 = bboxes[:,1]
        bbox_y1 = bboxes[:,2]
        bbox_y2 = bboxes[:,3]

        areas = (bbox_x2 - bbox_x1 + 1) * (bbox_y2 - bbox_y1 + 1)
        order = torch.argsort(conf_score, descending=True)

        while order.numel() > 0:
            if order.numel() == 1:
                idx_max_score = order.item()
            else:
                idx_max_score = order[0].item()
            
            # append the bbox with max score into result list
            res_bboxes_idx.append(idx_max_score)

            if order.numel() == 1:
                break
            
            # calc iou
            x1 = bbox_x1[order[1:]].clamp(min=bbox_x1[idx_max_score].item())
            x2 = bbox_x2[order[1:]].clamp(max=bbox_x2[idx_max_score].item())
            y1 = bbox_y1[order[1:]].clamp(min=bbox_y1[idx_max_score].item())
            y2 = bbox_y2[order[1:]].clamp(max=bbox_y2[idx_max_score].item())
            inter = (x2-x1).clamp(min=1) * (y2-y1).clamp(min=1)

            iou = inter /  (areas[idx_max_score]+areas[order[1:]]-inter)
            idx = (iou <= thresh).nonzero().squeeze()
            if idx.numel() == 0 :
                break
            order = order[idx+1]

    return torch.LongTensor(res_bboxes_idx)


def write_results(pred, confidence, num_classes, nms_thresh=0.4):
    conf_mask = (pred[:,:,4] > confidence).float().unsqueeze(2)
    pred = pred * conf_mask

    bbox_corner = pred.new(pred.shape)
    bbox_corner[:,:,0] = pred[:,:,0] - pred[:,:,2]/2
    bbox_corner[:,:,1] = pred[:,:,1] - pred[:,:,3]/2
    bbox_corner[:,:,3] = pred[:,:,0] + pred[:,:,2]/2
    bbox_corner[:,:,3] = pred[:,:,1] + pred[:,:,3]/2
    pred[:,:,:4] = bbox_corner[:,:,:4]

    bs = pred.size(0)
    write = False
    for idx in range(bs):
        each_img_pred = pred[idx] # BHW, 5+c
        non_zero_idx = each_img_pred[:,4].nonzero().squeeze()
        each_img_pred = each_img_pred[non_zero_idx,:]

        max_conf, pred_class = torch.max(each_img_pred[:,5:5+num_classes], 1)
        img_classes = pred_class.unique()

        max_conf = max_conf.float().unsqueeze(1)
        pred_class = pred_class.float().unsqueeze(1)
        seq = (each_img_pred[:,:5], max_conf, pred_class)
        each_img_pred = torch.cat(seq, 1)
        
        for cls in img_classes:
            each_cls_pred = each_img_pred[each_img_pred[:,-1] == cls, :]
            each_cls_pred_idx = nms(each_cls_pred[:,:4], each_cls_pred[:,4], nms_thresh)
            
            each_cls_pred = each_cls_pred[each_cls_pred_idx]
            
            batch_ind = each_cls_pred.new(each_cls_pred.size(0), 1).fill_(idx)
            seq = (batch_ind, each_cls_pred)

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    try:
        return output
    except:
        return 0

def padding_resize(img, size):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = size
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((size[1], size[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, size):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    img = cv2.resize(img, (size, size))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

