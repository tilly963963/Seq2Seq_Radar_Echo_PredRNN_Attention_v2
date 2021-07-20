import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import predrnn
import numpy as np
class Model(object):
    def __init__(self, configs):
        self.configs = configs
        print("self.configs=",self.configs)
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn': predrnn.RNN
            'interact_dst_predrnn': predict.DST_PredRNN
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.MSE_criterion = nn.MSELoss()

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
    def train(self, frames, mask):
        print("model ")
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        print("train")
        # print("frames_tensor",np.array(frames_tensor).shape)
#        frames_tensor (8, 20, 16, 16, 16)
#        mask_tensor (8, 9, 16, 16, 16)
        next_frames = self.network(frames_tensor, mask_tensor)
#        print("next_frames",np.array(next_frames).shape)
        
#        frames_tensor1=frames_tensor[:, 1:].detach().numpy()
#        print("frames_tensor1",np.array(frames_tensor1).shape) 
#        print("next_frames.dtype",next_frames.dtype)
        
#        next_frames1=next_frames.detach().numpy()
#        print("next_frames1",np.array(next_frames1).shape)   
#        print("frames_tensor[:, 1:].dtype",frames_tensor[:, 1:].dtype)
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
#        frames_tensor1=frames_tensor[:, 1:].detach().numpy()
#        print("frames_tensor1",np.array(frames_tensor1).shape)   
        loss.backward()
        self.optimizer.step()

        
    
        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()