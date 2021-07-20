
import torch
import torch.nn as nn
import gc
from torch.utils.checkpoint import checkpoint    

class InterDST_LSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm,r):
        super(InterDST_LSTMCell, self).__init__()
        self.r = r
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.layer_norm = nn.LayerNorm([num_hidden,width,width])
        self.c_norm = nn.LayerNorm([num_hidden, width, width])
        self.s_norm = nn.LayerNorm([num_hidden, width, width])

        self.c_attn_ = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),#.half()
            nn.LayerNorm([num_hidden, width, width]),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_hidden, kernel_size=1, stride=1, padding=0),
            # nn.LayerNorm([num_hidden, width, width]),
            # nn.ReLU(),
            # nn.LeakyReLU()
            # nn.Dropout2d(p=0.9)
        )
        self.s_attn_ = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden, width, width]),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.LayerNorm([num_hidden, width, width]),
            # nn.ReLU(),
            # nn.LeakyReLU()
            # nn.Dropout2d(p=0.9)
        )
        self.attn_ = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.LayerNorm([num_hidden, width, width]),
            # nn.Dropout2d(p=0.9)
            # nn.LeakyReLU()
        )
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 7, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden, width, width]),
            # nn.LeakyReLU()
        )
        self.conv_last = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            )#nn.LeakyReLU(negative_slope=0.001))

        
    
        '''
        self.conv_x_h = []
        self.conv_x_x = []
        self.conv_h_x = []
        self.conv_h_h = []

        for i in range(self.r):
            self.conv_x_h.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
                    nn.LayerNorm([num_hidden, width, width])
                )
            )
            self.conv_x_x.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=filter_size, stride=stride, padding=self.padding),
                    nn.LayerNorm([in_channel, width, width])
                )
            )
            self.conv_h_x.append(
                nn.Sequential(
                    nn.Conv2d(num_hidden, in_channel, kernel_size=filter_size, stride=stride, padding=self.padding),
                    nn.LayerNorm([in_channel, width, width])
                )
            )
            self.conv_h_h.append(
                nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
                    nn.LayerNorm([num_hidden, width, width])
                )
            )
        self.conv_x_h = nn.ModuleList(self.conv_x_h)
        self.conv_x_x = nn.ModuleList(self.conv_x_x)
        self.conv_h_x = nn.ModuleList(self.conv_h_x)
        self.conv_h_h = nn.ModuleList(self.conv_h_h)
        '''
    def _attn_channel(self,in_query,in_keys,in_values):
        print("_attn_channel()##")
        q_shape = in_query.shape
        k_shape = in_keys.shape
        batch = q_shape[0]
        num_channels = q_shape[1]
        width = q_shape[2]
        height = q_shape[3]
        length = k_shape[1]
        print("in_query=",in_query.size())
        print("in_keys=",in_keys.size())
        print("in_values=",in_values.size())

        gc.collect()
        torch.cuda.empty_cache()
        query = in_query.reshape([batch,num_channels,-1])
        key = in_keys.reshape([batch,-1,height*width]).permute((0, 2, 1))
        value = in_values.reshape([batch,-1,height*width]).permute((0, 2, 1))
        print("query=",query.size())
        print("keys=",key.size())
        print("values=",value.size())

        # query = query.cpu()#!
        # key = key.cpu()#!
        torch.cuda.empty_cache()
        attn = torch.matmul(query,key)
        attn = torch.nn.Softmax(dim=2)(attn)
        print("attn1=",attn.size())
        
        # attn = attn.cpu()#!!!!!
        # value = value.cpu()#!!!!!!
        # attn = attn.float()
        # value = value.float()

        # torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        attn = torch.matmul(attn,value.permute(0,2,1))
        print("attn2=",attn.size())
        # attn=attn.cuda()#!!!!!!!
        
        attn = attn.reshape([batch,num_channels,width,height])
        print("attn3=",attn.size())
        
        print("_attn_channel attn.is_cuda=",attn.is_cuda)
        torch.cuda.empty_cache()

        return attn

    def _attn_spatial(self,in_query,in_keys,in_values):
        print("_attn_spatial()##")
        # in_query = in_query.half()
        # in_keys = in_keys.half()
        # in_values = in_values.half()
        
        q_shape = in_query.shape
        k_shape = in_keys.shape
        batch = q_shape[0]
        num_channels = q_shape[1]
        width = q_shape[2]
        height = q_shape[3]
        length = k_shape[1]
        torch.cuda.empty_cache()
        print("in_query=",in_query.size())
        print("in_keys=",in_keys.size())
        print("in_values=",in_values.size())
        
        gc.collect()
        torch.cuda.empty_cache()
        query = in_query.reshape([batch,num_channels,-1]).permute((0,2,1))
        key = in_keys.permute((0,1,3,4,2)).reshape([batch,-1,num_channels])
        value = in_values.permute((0, 1, 3, 4, 2)).reshape([batch, -1, num_channels])
        # torch.cuda.memory_summary(device=None, abbreviated=False)
        # query = query.cpu()#!
        # key = key.cpu()#!
        torch.cuda.empty_cache()
        print("query=",query.size())
        print("keys=",key.size())
        print("values=",value.size())

        gc.collect()
        torch.cuda.empty_cache()
        attn = torch.matmul(query,key.permute(0,2,1))
        # attn = attn.cpu()#!!!!!!
        attn = torch.nn.Softmax(dim=2)(attn)
        print("attn1=",attn.size())

        # value = value.cpu()#!!!!!!!
        # attn = attn.float()
        # value = value.float()

        torch.cuda.empty_cache()
        # attn=attn.cuda()#!
        attn = torch.matmul(attn,value)
        print("attn2=",attn.size())
        # attn=attn.cuda()#!!!!!!
        
        attn = attn.reshape([batch,width,height,num_channels]).permute(0,3,1,2)
        print("tensors.device=",attn.device)
        print("attn3=",attn.size())
        torch.cuda.empty_cache()
        print("tensors.device=",attn.device)

        # mask_tensor = torch.FloatTensor(mask).to(self.configs.device)#!

        print("_attn_spatial attn.is_cuda=",attn.is_cuda)
        return attn

    def attn_sum_fussion(self,c ,in_query,in_keys,in_values):
        print("attn_sum_fussion()##")
        # aa=c + self._attn_spatial(in_query, in_keys, in_values)
        
        # aa = aa.float()
        gc.collect()
        torch.cuda.empty_cache()
        spatial_attn = self.s_norm(c + self._attn_spatial(in_query, in_keys, in_values))
        # bb=c + self._attn_channel(in_query, in_keys, in_values)
        # bb = bb.float()
       
        channel_attn = self.c_norm(c + self._attn_channel(in_query, in_keys, in_values))
        print("spatial_attn.size()=",spatial_attn.size())
        print("channel_attn.size()=",channel_attn.size())
        # print("cuda = ",config.gpu_options.allow_growth ,"gpu_list=",gpu_list)

        s_attn = self.s_attn_(spatial_attn)
        c_attn = self.c_attn_(channel_attn)
        attn = s_attn + c_attn

        attn = self.attn_(attn)
        # attn = attn + torch.zeros(1, dtype=attn.dtype, device=attn.device, requires_grad=True)
        
        return attn


        # mem = torch.cat((c_new, m_new), 1)
    def cat_m(self, c_new, m_new):
        mem = torch.cat((c_new, m_new), 1)
        return mem 
    def dot_m_new(self,f_t_prime ,m_t ,i_t_prime , g_t_prime):
        m_new = f_t_prime * m_t + i_t_prime * g_t_prime
        # mem = checkpoint(self.dot_m_new
        return m_new
    def forward(self, x_t, h_t, c_t,c_historys, m_t):
        # x_t=x_t.cuda()#!!!!!!
        # h_t=h_t.cuda()#!!!!!!
        # c_t=c_t.cuda()#!!!!!!
        # c_historys=c_historys.cuda()#!!!!!!
        # m_t=m_t.cuda()#!!!!!!


        print("forward!!!!!!!!!!!!")
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        # x_concat = x_concat.half()
        # h_concat = h_concat.half()
        # m_concat = m_concat.half()

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        gc.collect()
        torch.cuda.empty_cache()
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        gc.collect()
        torch.cuda.empty_cache()
        # torch.cuda.memory_summary(device=None, abbreviated=False)
        print("c_t.size()",c_t.size())
        print("f_t.size()",f_t.size())
        print("c_historys.size()",c_historys.size())
        print("i_t.size()",i_t.size())
        print("g_t.size()",g_t.size())

        # c_t = c_t.half()
        # f_t = f_t.half()
        # c_historys = c_historys.half()
        
        c_t = c_t + torch.zeros(1, dtype=c_t.dtype, device=c_t.device, requires_grad=True)
        f_t = f_t + torch.zeros(1, dtype=f_t.dtype, device=f_t.device, requires_grad=True)
        c_historys = c_historys + torch.zeros(1, dtype=c_historys.dtype, device=c_historys.device, requires_grad=True)
        
        c_new = checkpoint(self.attn_sum_fussion,c_t, f_t, c_historys, c_historys)+ i_t * g_t#?
        
        # c_new = self.attn_sum_fussion(c_t, f_t, c_historys, c_historys) + i_t * g_t###############
        print("c_new.size()",c_new.size())

        gc.collect()
        torch.cuda.empty_cache()

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        '''#?
        f_t_prime = f_t_prime + torch.zeros(1, dtype=f_t_prime.dtype, device=f_t_prime.device, requires_grad=True)
        m_t = m_t + torch.zeros(1, dtype=m_t.dtype, device=m_t.device, requires_grad=True)
        i_t_prime = i_t_prime + torch.zeros(1, dtype=i_t_prime.dtype, device=i_t_prime.device, requires_grad=True)
        g_t_prime = g_t_prime + torch.zeros(1, dtype=g_t_prime.dtype, device=g_t_prime.device, requires_grad=True)
        '''
        m_new = f_t_prime * m_t + i_t_prime * g_t_prime
        # m_new = checkpoint(self.dot_m_new,f_t_prime , m_t , i_t_prime ,g_t_prime)#?
        
        '''
        c_new = c_new + torch.zeros(1, dtype=c_new.dtype, device=c_new.device, requires_grad=True)
        m_new = m_new + torch.zeros(1, dtype=m_new.dtype, device=m_new.device, requires_grad=True)
        
        mem = checkpoint(self.cat_m,c_new,m_new)#!
        '''
        mem = torch.cat((c_new, m_new), 1)
        
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))

        h_new = o_t * torch.tanh(self.conv_last(mem))

        # h_new = h_new.half()
        # c_new = c_new.half()
        # h_new = self.do_LeakyReLU(h_new)
        gc.collect()
        torch.cuda.empty_cache()
        return h_new, c_new, m_new