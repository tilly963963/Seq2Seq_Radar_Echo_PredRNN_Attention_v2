__author__ = 'yunbo'

import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell_LayerNorm import SpatioTemporalLSTMCell
import numpy as np
import sys
class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            #第0層輸入圖片的channel是16, filter_num =64
#            第1,2,3層輸入channel是64 
#        in_channel= 16 num_hidden= 64 width= 16 layer_norm= 1
#        in_channel= 64 num_hidden= 64 width= 16 layer_norm= 1
#        in_channel= 64 num_hidden= 64 width= 16 layer_norm= 1
#        in_channel= 64 num_hidden= 64 width= 16 layer_norm= 1
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        print("cell_list=",np.array(cell_list).shape)
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, frames, mask_true):
# [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        next_frames = []
        h_t = []
        c_t = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
        for t in range(self.configs.total_length-1):#19
            if t < self.configs.input_length:#10
                net = frames[:,t]
            else:#9               
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)#h(t-1,l0) c(h-1,l0)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)#h(t,l-1) h(t-1,l) c(t-1,l) 
            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames1=next_frames
        print("next_frames1",np.array(next_frames1).shape)        
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        return next_frames