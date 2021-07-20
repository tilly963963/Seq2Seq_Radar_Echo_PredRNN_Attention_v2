__author__ = 'yunbo'

import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell_LayerNorm import SpatioTemporalLSTMCell
import numpy as np
import sys
import argparse


import torch
import tensorwatch as tw
import torchvision.models
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')
parser.add_argument('--input_length', type=int, default=6)
parser.add_argument('--total_length', type=int, default=12)
parser.add_argument('--img_width', type=int, default=512)
parser.add_argument('--img_channel', type=int, default=1)



# model
parser.add_argument('--model_name', type=str, default='predrnn')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=8)
parser.add_argument('--layer_norm', type=int, default=1)

# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=100)#50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.01)#0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)#0.001)
parser.add_argument('--reverse_input', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=5000)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

# print(args)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        configs = parser.parse_args()
        num_layers=4
        num_hidden=[64,64,64,64]
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


    def forward(self, frames):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        #frames_tensor (8, 20, 16, 16, 16)  #        mask_tensor (8, 9, 16, 16, 16)
        print("frames_tensor=",frames.shape)
        frames = frames.permute(0, 1, 4, 2, 3).contiguous().to(self.configs.device)
        # mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = torch.ones([1, 9, 64, 64,64]).to(self.configs.device)
        
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        next_frames = []
        h_t = []
        c_t = []
#        print("self.num_layers",self.num_layers)#4
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
        for t in range(self.configs.total_length-1):#19
#            print("self.configs.input_length=",self.configs.input_length)#10
            if t < self.configs.input_length:#10
                net = frames[:,t]
#                print("frames=>net",np.array(net.numpy() ).shape)# (8, 16, 16, 16)
            else:#9          
                    #  
                # net = frames[:,t]

                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen
#                net1=net.detach().numpy()
#                print("mask_true=>net",np.array(net1).shape)#(8, 16, 16, 16)
#            print(' ')
#            print("i=0")

            # with torch.no_grad():
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)#h(t-1,l0) c(h-1,l0)
#            h_t_0= h_t[0].detach().numpy()
#            c_t_0= c_t[0].detach().numpy()
#            memory_0=memory.detach().numpy()
#            print("h_t_0=",np.array(h_t_0).shape,"c_t_0=",np.array(c_t_0).shape,"memory=",np.array(memory_0).shape)
#            h_t_0= (8, 64, 16, 16) c_t_0= (8, 64, 16, 16) memory= (8, 64, 16, 16)
#            h_t_i=[]
#            c_t_i=[]
            
            for i in range(1, self.num_layers):
#                print(' ')
#                print("i=",i)
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)#h(t,l-1) h(t-1,l) c(t-1,l) 
#                print("h_t[i]=",h_t[i].size()," c_t[i]=", c_t[i].size(),"memory",memory.size())

         
                
#            sys.exit()
            x_gen = self.conv_last(h_t[self.num_layers-1])
#            x_gen1=x_gen.detach().numpy()
#            print("x_gen1",np.array(x_gen1).shape)#(8, 16, 16, 16)
            next_frames.append(x_gen)
        
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames1=next_frames
        print("next_frames1",np.array(next_frames1).shape)        
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        return next_frames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SimpleConv().to(device)
model2=RNN().to(device)
img = tw.draw_model(model2, [ 1,20, 64, 64, 64])
save_path ='picture_model/'
import os
if not os.path.isdir(save_path):
    os.makedirs(save_path)
    
img.save('predrnn.jpg')
# E:yu_ting/predrnn/predrnn_gogo/core/models