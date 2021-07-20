
import torch
import torch.nn as nn
from core.layers.InterDST_LSTMCell import InterDST_LSTMCell
import numpy as np
import sys
import gc

from typing import List, Tuple, Union
class InteractionDST_PredRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(InteractionDST_PredRNN, self).__init__()

        print("InterDST_LSTMCell=",InterDST_LSTMCell)
        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size


        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                InterDST_LSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                configs.stride, configs.layer_norm, configs.r
                               )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)



        print("self.cell_list=",self.cell_list)

    


    def list_shape(self, ndarray: Union[List, float]) -> Tuple[int, ...]:
        if isinstance(ndarray, list):
            # More dimensions, so make a recursive call
            outermost_size = len(ndarray)
            row_shape = ndarray[0].shape
            return (outermost_size, *row_shape)
        else:
            # No more dimensions, so we're done
            return ()
    def forward(self, frames, is_training=True):
    # def forward(self, frames, mask_true, is_training=True):

        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        # mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()#!

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []
        c_t_history = []
        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            c_t_history.append(zeros.unsqueeze(1))#!在第一個維度多一個維度(?)
        memory = torch.zeros([batch, self.num_hidden[-1], height, width]).cuda()
        print("type(memory)=", memory.dtype)

        z_t = torch.zeros([batch, self.num_hidden[0], height,
                           width],dtype=torch.float16).cuda()
        print("type(z_t)=",z_t.dtype)
        # print("self.cell_list=",self.cell_list)
        # print("init => h_t=",np.array(h_t).shape)#(4,)
        # print("c_t=",np.array(c_t).shape)#(4,)
        # print("c_t_history=",np.array(c_t_history).shape)#(4,)
        # print("memory=",memory.size(),"z_t=",z_t.size())
    
        # memory= torch.Size([1, 64, 64, 64]) z_t= torch.Size([1, 64, 64, 64])
        # if is_training:#!
        seq_length = self.configs.total_length
        # else:
        #     seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):
                
            gc.collect()
            torch.cuda.empty_cache()
            print("##### t=",t," #####")
            net = frames[:, t]
            '''
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen
            '''
            print("type(net)=",net.dtype)
            print("type(net)=",h_t[0].dtype)
            # print("layers = 1 input h_t[0]=",h_t[0].size())
            # print("c_t[0]=",c_t[0].size(),"c_t_history[0]=",c_t_history[0].size())
            # print("memory=",memory.size())
            # layers = 1 input h_t[0]= torch.Size([1, 64, 64, 64])
            # c_t[0]= torch.Size([1, 64, 64, 64]) 
            #!t=0 c_t_history[0]= torch.Size([1, 1, 64, 64, 64])
            #!t=1 c_t_history[0]= torch.Size([1, 2, 64, 64, 64])
            #!t=10 c_t_history[0]= torch.Size([1, 11, 64, 64, 64])
            # memory= torch.Size([1, 64, 64, 64])

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0],c_t_history[0], memory)
            # print("layers = 1 output h_t[0]=",h_t[0].size(),"c_t[0]=",c_t[0].size(),"memory=",memory.size())
            # layers = 1 output h_t[0]= torch.Size([1, 64, 64, 64]) c_t[0]= torch.Size([1, 64, 64, 64]) memory= torch.Size([1, 64, 64, 64])

            # print("type(memory)=", memory.dtype,"h_t[0]=",h_t[0].dtype,"")
            c_t_history[0] = torch.cat([c_t_history[0],c_t[0].unsqueeze(1)],1)
            # print("c_t_history[0] cat =",c_t_history[0].size())
            #! t=0 c_t_history[0] cat = torch.Size([1, 2, 64, 64, 64])
            #! t=1 c_t_history[0] cat = torch.Size([1, 3, 64, 64, 64])
            #! t=10 c_t_history[0] cat = torch.Size([1, 12, 64, 64, 64])
            print("c_t_history =",self.list_shape(c_t_history))
            # c_history[0] = c_history[0].half()

            gc.collect()
            torch.cuda.empty_cache()            
            for i in range(1, self.num_layers):
                print("---- i=",i,"----")
                # print(" input h_t[i-1]=",h_t[i-1].size(),"h_t[i]=",h_t[i].size(),
                # "c_t[i]=",c_t[i].size(),"c_t_history[i]=",c_t_history[i].size(),"memory=",memory.size())
# input h_t[i-1]= torch.Size([1, 64, 64, 64]) h_t[i]= torch.Size([1, 64, 64, 64]) 
# c_t[i]= torch.Size([1, 64, 64, 64]) 
#!t=0 i=1~4 c_t_history[1~3]= torch.Size([1, 1, 64, 64, 64]) 
#!t=1 i=1~4 c_t_history[1~3]= torch.Size([1, 2, 64, 64, 64])
#!t=10 i=1~4 c_t_history[1~3]= torch.Size([1, 11, 64, 64, 64])
# memory= torch.Size([1, 64, 64, 64])
                # import torch
                # import gc
                gc.collect()
                torch.cuda.empty_cache()
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],c_t_history[i],
                                                                memory)
                # import gc
                gc.collect()
                torch.cuda.empty_cache()
                # c_t_history[i] = c_t_history[i].cpu()#!!!!!!!
                # c_t[i] = c_t[i].cpu()#!!!!!!!

                c_t_history[i] = torch.cat([c_t_history[i], c_t[i].unsqueeze(1)], 1)
                # h_t[i] = h_t[i].float()
                # c_t[i] = c_t[i].float()
                # c_t_history[i] = c_t_history[i].float()
                # memory = memory.float()
                print("c_t_history=",self.list_shape(c_t_history))
                #c_t_history= (4, 1, 12, 64, 64, 64)
                # print(" output h_t[i]=",h_t[i].size(),"c_t[i]=",c_t[i].size(),"memory=",memory.size(),"c_t_history[i]=",c_t_history[i].size())
# _attn_spatial
# attn.is_cuda= True
# tensors.device= cuda:0
# output h_t[i]= torch.Size([1, 64, 64, 64]) c_t[i]= torch.Size([1, 64, 64, 64]) 
# memory= torch.Size([1, 64, 64, 64]) 
#!t=0 i=1~4 c_t_history[i]= torch.Size([1, 2, 64, 64, 64])
#!t=1 i=1~4 c_t_history[0]= torch.Size([1, 3, 64, 64, 64])
#!t=10 i=1~4 c_t_history[0]= torch.Size([1, 12, 64, 64, 64])
            print("h_t =",self.list_shape(h_t))
            gc.collect()
            torch.cuda.empty_cache()    
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            # print("h_t[self.num_layers - 1]=",h_t[self.num_layers - 1].size())
            print("x_gen=",x_gen.size())
# x_gen= torch.Size([1, 64, 64, 64])
            next_frames.append(x_gen)
            print("np.array(next_frames).shape = ",np.array(next_frames).shape)
# np.array(next_frames).shape =  (11,)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        gc.collect()
        torch.cuda.empty_cache()
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4,
                                                              2).contiguous()
        print("next_frames.shape = ",next_frames.size())
# next_frames.shape =  torch.Size([1, 11, 64, 64, 64])
        # nexts_frames=next_frames.float16()
        # next_frames=next_frames.half()#.float()
        return next_frames