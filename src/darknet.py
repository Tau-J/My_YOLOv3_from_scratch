from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

def parse_cfg(cfgfile):
    """
    Takes a configuration file 
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0]        # Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()
    prev_filters = 3            # previous feature map is an image, so the number of filters is 3 (R, G, B)
    output_filters = []

    for idx, each_block in enumerate(blocks[1:]):
        module = nn.Sequential()
        # check the type of block
        # create a new module for the block
        # append to module_list
        if each_block['type'] == 'convolutional':
            try:
                bn = int(each_block['batch_normalize'])
                bias = False
            except:
                bn = 0
                bias = True
            filters = int(each_block['filters'])
            size = int(each_block['size'])
            stride = int(each_block['stride'])
            pad = int(each_block['pad'])
            activation = each_block['activation']

            if pad:
                pad = (size - 1) // 2
            else:
                pad = 0

            # add conv layer
            conv = nn.Conv2d(prev_filters, filters, size, stride, pad, bias=bias)
            module.add_module('conv_{}'.format(idx), conv)

            # add bn layer
            if bn:
                bn = nn.BatchNorm2d(filters)
                module.add_module('bn_{}'.format(idx), bn)
            
            # check the activation
            # activation will be either leaky or linear
            if activation == 'leaky':
                leaky = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{}'.format(idx), leaky)
            
        elif each_block['type'] == 'upsample':
            stride = int(each_block['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module("upsample_{}".format(idx), upsample)

        elif each_block['type'] == 'route':
            layers = each_block['layers'].split(',')
            start = int(layers[0])
            try:
                end = int(layers[1])
            except:
                end = 0

            if start > 0:
                start = start - idx 
            if end > 0:
                end = end - idx     # a trick to let end negative, to keep end + idx correct
            
            route =  EmptyLayer()
            module.add_module("route_{0}".format(idx), route)

            if end < 0:
                end = output_filters[end + idx]
            filters = output_filters[start + idx] + end

        elif each_block['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(idx), shortcut)

        elif each_block['type'] == 'yolo':
            mask = each_block['mask'].split(',')
            mask = list(map(lambda x: int(x), mask))

            anchors = each_block['anchors'].split(',')
            anchors = list(map(lambda x: int(x), anchors))
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('detection_{}'.format(idx), detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

# blocks = parse_cfg("cfg/yolov3.cfg")
# print(create_modules(blocks))

class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)
