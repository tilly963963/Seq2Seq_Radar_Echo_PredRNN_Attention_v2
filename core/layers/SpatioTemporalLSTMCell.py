__author__ = 'yunbo'

import torch
import torch.nn as nn
import numpy as np
class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()
        print("in_channel=",in_channel,"num_hidden=",num_hidden,"width=",width,"layer_norm=",layer_norm)
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
#        in_channel= 16 num_hidden= 64 width= 16 layer_norm= 1
#        in_channel= 64 num_hidden= 64 width= 16 layer_norm= 1
#        in_channel= 64 num_hidden= 64 width= 16 layer_norm= 1
#        in_channel= 64 num_hidden= 64 width= 16 layer_norm= 1
#        cell_list= (4,)
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
        #     nn.LayerNorm([num_hidden * 7, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
        #     nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
        #     nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
        #     nn.LayerNorm([num_hidden, width, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

#LayerNorm：channel方向做归一化，算CHW的均值，主要对RNN作用明显；https://blog.csdn.net/shanglianlm/article/details/85075706
    def forward(self, x_t, h_t, c_t, m_t):
#        print("forward SpatioTemporalLSTMCell")
#        print("x_t",x_t.size())
#        print("h_t",h_t.size())
#        print("c_t",c_t.size())
#        print("m_t",m_t.size())
#        x_t (8, 16, 16, 16)
#        h_t (8, 64, 16, 16)
#        c_t (8, 64, 16, 16)
#        m_t (8, 64, 16, 16)
#        print("上一步驟做h和m做conv x也做conv---")
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        
#        print("x_concat",x_concat.size())
#        print("h_concat",h_concat.size())
#        print("m_concat",m_concat.size())
#        x_concat torch.Size([8, 448, 16, 16])
#        h_concat torch.Size([8, 256, 16, 16])
#        m_concat torch.Size([8, 192, 16, 16])
#        print("x分割---")
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
#        print("i_x",i_x.size())
#        print("f_x",f_x.size())
#        print("g_x",g_x.size())
#        print("i_x_prime",i_x_prime.size())
#        print("f_x_prime",f_x_prime.size())
#        print("g_x_prime",g_x_prime.size())        
#        print("o_x",o_x.size())        
#        print("h分割---")
#        i_x torch.Size([8, 64, 16, 16])
#        f_x torch.Size([8, 64, 16, 16])
#        g_x torch.Size([8, 64, 16, 16])
#        i_x_prime torch.Size([8, 64, 16, 16])
#        f_x_prime torch.Size([8, 64, 16, 16])
#        g_x_prime torch.Size([8, 64, 16, 16])
#        o_x torch.Size([8, 64, 16, 16])
        
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)#T-1
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)
#        print("i_h",i_h.size(), "f_h",f_h.size(),"g_h",g_h.size(),"o_h",o_h.size())
#        i_h torch.Size([8, 64, 16, 16]) f_h torch.Size([8, 64, 16, 16]) g_h torch.Size([8, 64, 16, 16]) o_h torch.Size([8, 64, 16, 16])
#        print("m分割---")
#        print("i_m",i_m.size(), "f_m",f_m.size(),"g_m",g_m.size())
#        i_m torch.Size([8, 64, 16, 16]) f_m torch.Size([8, 64, 16, 16]) g_m torch.Size([8, 64, 16, 16])
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t
        
#        print("t上面---")
#        print("i_t",i_t.size())
#        print("f_t",f_t.size())
#        print("g_t",g_t.size())        
#        print("c_new",c_new.size())  
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
#        i_t torch.Size([8, 64, 16, 16])
#        f_t torch.Size([8, 64, 16, 16])
#        g_t torch.Size([8, 64, 16, 16])
#        c_new torch.Size([8, 64, 16, 16])


        m_new = f_t_prime * m_t + i_t_prime * g_t_prime
#        print("t下面---")
#        print("i_t_prime",i_t_prime.size())
#        print("f_t_prime",f_t_prime.size())
#        print("g_t_prime",g_t_prime.size())        
#        print("m_new",m_new.size()) 
#        i_t_prime torch.Size([8, 64, 16, 16])
#        f_t_prime torch.Size([8, 64, 16, 16])
#        g_t_prime torch.Size([8, 64, 16, 16])
#        m_new torch.Size([8, 64, 16, 16])
        
        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
#        print("輸出---")
#        print("mem",mem.size())
#        print("o_t",o_t.size())        
#        print("h_new",h_new.size()) 
#        mem torch.Size([8, 128, 16, 16])
#        o_t torch.Size([8, 64, 16, 16])
#        h_new torch.Size([8, 64, 16, 16])
#        h_t_0= (8, 64, 16, 16) c_t_0= (8, 64, 16, 16) memory= (8, 64, 16, 16)
        return h_new, c_new, m_new









