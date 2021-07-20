import os
import torch
import torch.nn as nn
from torch.optim import Adam
# from core.models import predrnn
from core.models import predrnn_LayerNorm,predict,predict_checkpoint
import numpy as np
# from numba import cuda
import gc
# from apex import amp
# from apex.fp16_utils import *
from torch.cuda.amp import autocast, GradScaler
# from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import autocast as autocast
from torch.autograd import Variable
class MyMSELoss(torch.nn.Module):
  def __init__(self, weight):
    super(MyMSELoss, self).__init__()
    self.weight =  weight
   
  def forward(self, output, label):
    print("==================")
    # error = output - label#!
    label=label.float() 
    error = label - output#!
    print(output - label)
    '''
    依照label的dBz區間 給予誤差不同權重
    '''
    '''
    error_weight = torch.where((label > 45) & (label <= 65), error*self.weight[0], error)

    error_weight = torch.where((label > 40) & (label <= 45), error*self.weight[1], error_weight)

    error_weight = torch.where((label > 35) & (label <= 40), error*self.weight[2], error_weight)
    error_weight = torch.where((label > 30)& (label <= 35), error_weight*self.weight[3], error_weight)   
    # error_weight = torch.where((label > 25) &(label <= 30), error_weight*self.weight[4], error_weight)
    error_weight = torch.where((label > 0)& (label <= 1), error_weight*self.weight[4], error_weight)   
    '''
    # error_weight = torch.where((label <= 5), torch.pow(error,2)*self.weight[0], error)
    
    error_weight = torch.where((label < 22), torch.pow(error,2)*self.weight[0], error)

    error_weight = torch.where((label >= 22) & (label < 28), torch.pow(error,2)*self.weight[1], error_weight)

    error_weight = torch.where((label >= 28) & (label < 33), torch.pow(error,2)*self.weight[2], error_weight)
    error_weight = torch.where((label >= 33) & (label < 40), torch.pow(error,2)*self.weight[3], error_weight)   
    # error_weight = torch.where((label > 25) &(label <= 30), error_weight*self.weight[4], error_weight)
    error_weight = torch.where((label >= 40) & (label < 45), torch.pow(error,2)*self.weight[4], error_weight) 
    '''
    error_weight = torch.where((label >= 45), torch.pow(error,2)*self.weight[5], error_weight) 
   '''
    error_weight = torch.where((label >= 45) & (label < 50), torch.pow(error,2)*self.weight[5], error_weight) 
    error_weight = torch.where((label >=50) , torch.pow(error,2)*self.weight[6], error_weight) 
   
    # error_weight =error_weight.half()#!

        # self.weight = [1,2,5,10,30]#!

# 1, x < 2
# 2, 2 ≤ x < 5
# 5, 5 ≤ x < 10
# 10, 10 ≤ x < 30
# 30, x ≥ 30
# '''
# print("2 rainfall_to_dBZ=",rainfall_to_dBZ(2))#22
# print("5 rainfall_to_dBZ=",rainfall_to_dBZ(5))#28
# print("10 rainfall_to_dBZ=",rainfall_to_dBZ(10))#33
# print("30 rainfall_to_dBZ=",rainfall_to_dBZ(30))#40
    print("加權後",error_weight)
    print("加權後 大小",error_weight.size())

    # error_weight = torch.pow(error_weight,2)
    # print("平方",error_weight)
  
    error_weight_mean = torch.mean(error_weight)
    print("avg=",error_weight_mean)
    
    # error_weight_mean = torch.sqrt(error_weight_mean)#?
    # error_weight_mean =error_weight_mean#.half()#!
    # print("sqrt=",error_weight_mean)

    return error_weight_mean
class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        print("self.configs=",self.configs)
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        # print("networks_map=",networks_map)
        networks_map = {
            'predrnn': predrnn_LayerNorm.RNN,
            'interact_dst_predrnn': predict.InteractionDST_PredRNN,
            'InterDST_LSTMCell_checkpoint':predict_checkpoint.InteractionDST_PredRNN
        }

        if configs.model_name in networks_map:
            print("configs.model_name=",configs.model_name)
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        print("self.network=",self.network.named_parameters())
        need_frozen_list=["cell_list.0.layer_norm.weight","cell_list.1.layer_norm.weight","cell_list.0.layer_norm.bias"
        ,"cell_list.0.c_norm.weight","cell_list.0.c_norm.bias","cell_list.0.s_norm.weight","cell_list.0.s_norm.bias","cell_list.0.c_attn_.0.weight","cell_list.0.c_attn_.0.bias"
        ,"cell_list.0.c_attn_.1.weight"
        ,"cell_list.0.c_attn_.1.bias"
        ,"cell_list.0.c_attn_.3.weight"
        ,"cell_list.0.c_attn_.3.bias"
        ,"cell_list.0.s_attn_.0.weight"
        ,"cell_list.0.s_attn_.1.weight"
        ,"cell_list.0.s_attn_.1.bias"
        ,"cell_list.0.s_attn_.3.weight"
        ,"cell_list.0.attn_.0.weight"
        ,"cell_list.0.conv_x.0.weight"
        ,"cell_list.0.conv_x.1.weight"
        ,"cell_list.0.conv_x.1.bias"
        ,"cell_list.0.conv_h.0.weight"
        ,"cell_list.0.conv_h.1.weight"
        ,"cell_list.0.conv_h.1.bias"
        ,"cell_list.0.conv_m.0.weight"
        ,"cell_list.0.conv_m.1.weight"
        ,"cell_list.0.conv_m.1.bias"
        ,"cell_list.0.conv_o.0.weight"
        ,"cell_list.0.conv_o.1.weight"
        ,"cell_list.0.conv_o.1.bias"


        ,"cell_list.1.layer_norm.bias"
        ,"cell_list.1.c_norm.weight"
        ,"cell_list.1.c_norm.bias"
        ,"cell_list.1.s_norm.weight"
        ,"cell_list.1.s_norm.bias"
        ,"cell_list.1.c_attn_.0.weight"
        ,"cell_list.1.c_attn_.0.bias"
        ,"cell_list.1.c_attn_.1.weight"
        ,"cell_list.1.c_attn_.1.bias"
        ,"cell_list.1.c_attn_.3.weight"
        ,"cell_list.1.c_attn_.3.bias"
        ,"cell_list.1.s_attn_.0.weight"
        ,"cell_list.1.s_attn_.1.weight"
        ,"cell_list.1.s_attn_.1.bias"
        ,"cell_list.1.s_attn_.3.weight"
        ,"cell_list.1.attn_.0.weight"
        ,"cell_list.1.conv_x.0.weight"
        ,"cell_list.1.conv_x.1.weight"
        ,"cell_list.1.conv_x.1.bias"
        ,"cell_list.1.conv_h.0.weight"
        ,"cell_list.1.conv_h.1.weight"
        ,"cell_list.1.conv_h.1.bias"
        ,"cell_list.1.conv_m.0.weight"
        ,"cell_list.1.conv_m.1.weight"
        ,"cell_list.1.conv_m.1.bias"
        ,"cell_list.1.conv_o.0.weight"
        ,"cell_list.1.conv_o.1.weight"
        ,"cell_list.1.conv_o.1.bias"
        ,"cell_list.1.conv_last.0.weight"

        ,"cell_list.2.layer_norm.weight"
        ,"cell_list.2.layer_norm.bias"
        ,"cell_list.2.c_norm.weight"
        ,"cell_list.2.c_norm.bias"
        ,"cell_list.2.s_norm.weight"
        ,"cell_list.2.s_norm.bias"
        ,"cell_list.2.c_attn_.0.weight"
        ,"cell_list.2.c_attn_.0.bias"
        ,"cell_list.2.c_attn_.1.weight"
        ,"cell_list.2.c_attn_.1.bias"
        ,"cell_list.2.c_attn_.3.weight"
        ,"cell_list.2.c_attn_.3.bias"
        ,"cell_list.2.s_attn_.0.weight"
        ,"cell_list.2.s_attn_.1.weight"
        ,"cell_list.2.s_attn_.1.bias"
        ,"cell_list.2.s_attn_.3.weight"
        ,"cell_list.2.attn_.0.weight"
        ,"cell_list.2.conv_x.0.weight"
        ,"cell_list.2.conv_x.1.weight"
        ,"cell_list.2.conv_x.1.bias"
        ,"cell_list.2.conv_h.0.weight"
        ,"cell_list.2.conv_h.1.weight"
        ,"cell_list.2.conv_h.1.bias"
        ,"cell_list.2.conv_m.0.weight"
        ,"cell_list.2.conv_m.1.weight"
        ,"cell_list.2.conv_m.1.bias"
        ,"cell_list.2.conv_o.0.weight"
        ,"cell_list.2.conv_o.1.weight"
        ,"cell_list.2.conv_o.1.bias"
        ,"cell_list.2.conv_last.0.weight"
        ,"cell_list.3.layer_norm.weight"
        ,"cell_list.3.layer_norm.bias"
        ,"cell_list.3.c_norm.weight"
        ,"cell_list.3.c_norm.bias"
        ,"cell_list.3.s_norm.weight"
        ,"cell_list.3.s_norm.bias"
        ,"cell_list.3.c_attn_.0.weight"
        ,"cell_list.3.c_attn_.0.bias"
        ,"cell_list.3.c_attn_.1.weight"
        ,"cell_list.3.c_attn_.1.bias"
        ,"cell_list.3.c_attn_.3.weight"
        ,"cell_list.3.c_attn_.3.bias"
        ,"cell_list.3.s_attn_.0.weight"
        ,"cell_list.3.s_attn_.1.weight"
        ,"cell_list.3.s_attn_.1.bias"
        ,"cell_list.3.s_attn_.3.weight"
        ,"cell_list.3.attn_.0.weight"
        ,"cell_list.3.conv_x.0.weight"
        ,"cell_list.3.conv_x.1.weight"
        ,"cell_list.3.conv_x.1.bias"
        ,"cell_list.3.conv_h.0.weight"
        ,"cell_list.3.conv_h.1.weight"
        ,"cell_list.3.conv_h.1.bias"
        ,"cell_list.3.conv_m.0.weight"
        ,"cell_list.3.conv_m.1.weight"
        ,"cell_list.3.conv_m.1.bias"]
        # ,"cell_list.3.conv_o.0.weight"
        # ,"cell_list.3.conv_o.1.weight"
        # ,"cell_list.3.conv_o.1.bias"
        # ,"cell_list.3.conv_last.0.weight"]
        
        for param in self.network.named_parameters():
            print("param=",param[0])
            if param[0] in need_frozen_list:
                print("is need_frozen_list",param[0])
                param[1].requires_grad = False
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=configs.lr)
        
        # self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        # self.MSE_criterion = nn.MSELoss()#!
        # self.weight = [40,30,15,10,5]
        # self.weight = [10,5,3,2]#!
        # self.weight = [8,3,2,1.5,0]#!
        # self.weight = [1,1.5,3,4,8,10]#!
        # self.weight = [1,2,3,5,10,30]#!
        # self.weight = [1,2,5,10,30,50]#!
        self.weight = [1,2,5,10,20,50,70]#!

        # self.weight = [1,1,2,3,8,15]#!
        # self.weight = [1,1.5,2,3,4,15]#!

      
        self.custom_criterion = MyMSELoss(self.weight)#!
        # self.network, self.optimizer = amp.initialize(self.network, self.optimizer, opt_level="O2")#custom_criterion=O2 
        # self.scaler = torch.cuda.amp.GradScaler()
        self.accumulation_steps = 2
        self.i = 0
    def save(self, model_name,save_path):
        stats = {}
        stats['net_param'] = self.network.state_dict()
#        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        
#        checkpoint_path = os.path.join(self.configs.save_dir, 'model.pkl'+'-'+str(itr))
#        torch.save(stats, checkpoint_path)
#        model_name = 'mode_haveLayerNorm_2y3m_itr{}'.format(itr)
        save_path = os.path.join(save_path,'{}'.format(model_name))
        torch.save(stats, save_path)
        print("save model to %s" % save_path)

#    def load(self, checkpoint_path):
#        print('load model:', checkpoint_path)
#        stats = torch.load(checkpoint_path)
#        self.network.load_state_dict(stats['net_param'])
    def load(self, save_path, model_name):
        save_path = os.path.join(save_path, model_name)
        print('load model:', save_path)
        stats = torch.load(save_path)
        self.network.load_state_dict(stats['net_param'])
#     def train(self, frames, mask):
#     # def train(self, frames):
#         # print(frames)
#         # frames = torch.from_numpy(frames)
#         # print("model  type(frames)=",frames.dtype)
#         frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
#         frames_tensor =frames_tensor.half()
#         print("type(frames_tensor)=",frames_tensor.dtype)
#         # mask_tensor = torch.ones([1, 9, 16, 16,16]).to(self.configs.device)
        
#         mask_tensor = torch.FloatTensor(mask).to(self.configs.device)#!
#         mask_tensor =mask_tensor.half()#!
#         print("type(mask_tensor)=",mask_tensor.dtype)
        
#         # self.optimizer.zero_grad()#!
# #        print("train")
#         # print("frames_tensor",np.array(frames_tensor).shape)
# #        frames_tensor (8, 20, 16, 16, 16)
# #        mask_tensor (8, 9, 16, 16, 16)

#         # frames_tensor = Variable(frames_tensor, requires_grad=True)

#         # with torch.no_grad():
#         # if torch.cuda.is_available():
#             # a = torch.rand([3,3]).cuda()
#             # frames_tensor=frames_tensor.cuda()
#         # next_frames = self.network(frames_tensor)
#         with autocast():#!
#             next_frames = self.network(frames_tensor, mask_tensor)
#             next_frames=next_frames.half()
#             print("type(next_frames)=",next_frames.dtype)

#     #        print("next_frames",np.array(next_frames).shape)
            
#     #        frames_tensor1=frames_tensor[:, 1:].detach().numpy()
#     #        print("frames_tensor1",np.array(frames_tensor1).shape) 
#     #        print("next_frames.dtype",next_frames.dtype)
            
#     #        next_frames1=next_frames.detach().numpy()
#     #        print("next_frames1",np.array(next_frames1).shape)   
#     #        print("frames_tensor[:, 1:].dtype",frames_tensor[:, 1:].dtype)
        

    
#             # loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])#!
#             loss = self.custom_criterion(next_frames, frames_tensor[:, 1:])#!
#             print("use rmse")
#             print("self.accumulation_steps")
#     #        
#     # frames_tensor1=frames_tensor[:, 1:].detach().numpy()
#     #        print("frames_tensor1",np.array(frames_tensor1).shape)   
#             # loss = torch.sqrt(loss)#?
#             # loss[loss > 5] = loss[loss* 2]
#             print("loss.size()=",loss.size())
#             loss=loss.float()
#             print("not loss def",loss)
#         # if loss>5:#?
#             # loss=loss.clone()*2#?
#         # loss[loss>5] = loss[loss>5].clone()
# # for i, j, k in zip(X, Y, Z):
#     # B[:, i, j] = A[:, i, j] + k
#         # with torch.autograd.set_detect_anomaly(True):

#             # loss[loss>5]=loss[loss>5].clone()*2
#             # print("loss def",loss)#?

#             loss = loss/self.accumulation_steps

#         # loss.backward()#!
#         # self.scaler.scale(loss).backward()#!
#         a=1
#         # self.optimizer.zero_grad()   # reset gradient#!

#         # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
#         self.scaler.scale(loss).backward()#梯度自动缩放
#             # a=0
#         print("loss=",loss)
#         print('梯度自动缩放',a)
#         # if a==1:
#         #     print('no 梯度自动缩放',a)

#         # loss.backward()#!

#         # self.optimizer.step()#!
# # clipping_value = 1 # arbitrary value of your choosing
# # torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
#         print("loss backward,self.i=",self.i)
#         # self.optimizer.step()#!
#         if((self.i+1)%self.accumulation_steps)==0:
#             # optimizer the net
#             print("optimizer",self.i+1)
#             # self.optimizer.step()        # update parameters of net#!
#             # self.optimizer.zero_grad()   # reset gradient#!
#             # self.scaler.step(optimizer)#!

#             # 准备着，看是否要增大scaler
#             # self.scaler.update()#!
#             '''
#             self.scaler.step(self.optimizer)
#             self.scaler.update()
#             self.optimizer.zero_grad()   # reset gradient#!
#             '''
#             self.scaler.unscale_(self.optimizer)
            
#             torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=2)

#             self.scaler.step(self.optimizer)
#             self.scaler.update()
#             self.optimizer.zero_grad()   # reset gradient#!
            
#             # self.i=0
#             # scheduler.step()
#         self.i=self.i+1
#         del frames_tensor
#         del next_frames
#         del mask_tensor
#         print("1234")
#         # from numba import cuda
#         # cuda.select_device(0)
#         # cuda.close()
#         # cuda.select_device(0)

#         # gc.collect()
#         # torch.cuda.empty_cache()

#         return loss.detach().cpu().numpy()
    def train(self, frames, mask):
        print("model ")
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        # self.optimizer.zero_grad()
#        print("train")
        # print("frames_tensor",np.array(frames_tensor).shape)
#        frames_tensor (8, 20, 16, 16, 16)
#        mask_tensor (8, 9, 16, 16, 16)

        # frames_tensor = Variable(frames_tensor, requires_grad=True)

        # with torch.no_grad():
        # if torch.cuda.is_available():
            # a = torch.rand([3,3]).cuda()
            # frames_tensor=frames_tensor.cuda()
        # with autocast():#####
        next_frames = self.network(frames_tensor, mask_tensor)
#        print("next_frames",np.array(next_frames).shape)
        
#        frames_tensor1=frames_tensor[:, 1:].detach().numpy()
#        print("frames_tensor1",np.array(frames_tensor1).shape) 
#        print("next_frames.dtype",next_frames.dtype)
        
#        next_frames1=next_frames.detach().numpy()
#        print("next_frames1",np.array(next_frames1).shape)   
#        print("frames_tensor[:, 1:].dtype",frames_tensor[:, 1:].dtype)
     
        # loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        loss = self.custom_criterion(next_frames, frames_tensor[:, 1:])#!
        print("use custom_criterion")
        print("loss=",loss)       
# frames_tensor1=frames_tensor[:, 1:].detach().numpy()
#        print("frames_tensor1",np.array(frames_tensor1).shape)   
        loss = loss/self.accumulation_steps
        
        loss.backward()
        if((self.i+1)%self.accumulation_steps)==0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        # self.scaler.scale(loss).backward()###

        # 将梯度值缩放回原尺度后，优化器进行一步优化
        # self.scaler.step(self.optimizer)###

        # 更新scalar的缩放信息
        # self.scaler.update()###
        self.i = self.i+1
        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        self.i=0
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        frames_tensor = Variable(frames_tensor, requires_grad=False)
        # with autocast():#!
        with torch.no_grad():
            next_frames = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()