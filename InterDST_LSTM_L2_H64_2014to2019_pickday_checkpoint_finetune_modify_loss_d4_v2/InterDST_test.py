# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:27:23 2020

@author: tilly963
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import timedelta

## ML lib
# import tensorflow as tf
# import keras.backend as K
# from keras.models import load_model
# from sklearn.externals import joblib

from visualize.Verification import Verification 

#from model.CRNN.ConvLSTM_v2 import ConvLSTM
# from data.radar_echo_NWP import load_data
# from data.radar_echo_CREF_p20_out315_0824 import load_data_CREF
import time
from data.radar_echo_p20_muti_sample_drop_08241800_load_512x512 import load_data

# from CustomUtils_v2 import SaveSummary

import os
import shutil
import argparse
import numpy as np
import torch
#from core.data_provider import datasets_factory
from core.models.model_factory_LayerNormpy import Model
from core.utils import preprocess
#import core.trainer as trainer

#for test train.py
import os.path
import datetime
# import cv2
import numpy as np
# from skimage.measure import compare_ssim
from core.utils import preprocess
from visualize.visualized_pred import visualized_area_with_map,visualized_area_with_map_mae
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

# data
parser.add_argument('--dataset_name', type=str, default='mnist')
#parser.add_argument('--train_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-train.npz')
#parser.add_argument('--valid_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--save_dir', type=str, default='checkpoints/try_mnist_predrnn')
parser.add_argument('--gen_frm_dir', type=str, default='results/try_mnist_predrnn')
parser.add_argument('--input_length', type=int, default=6)
parser.add_argument('--total_length', type=int, default=12)
parser.add_argument('--img_width', type=int, default=512)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='InterDST_LSTMCell_checkpoint')#InterDST_LSTMCell_checkpoint
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
parser.add_argument('--r', type=int, default=2)
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

args = parser.parse_args()
print(args)

a=args.img_width

    
def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:#50000
        eta -= args.sampling_changing_rate#1=1-0.00002
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))#(8, 9) 0~1的亂數
#    print("random_flip=",np.array(random_flip).shape)# (8, 9)
#    print("random_flip[0]=",random_flip[0])
#    random_flip[0]= [0.17844186 0.84035211 0.25251073 0.17387313 0.99843577 0.63203244
#     0.33349962 0.95222449 0.04404661]    
    true_token = (random_flip < eta)# (8, 9) 9預測的時間(T數量)
    
#    print("true_token",true_token)#(8, 9) 0 or 1
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = real_input_flag.astype('float16')
    print("real_input_flag1 =",real_input_flag.shape)# (72, 16, 16, 16)
    real_input_flag = np.reshape(real_input_flag,
                           (args.batch_size,
                            args.total_length - args.input_length - 1,
                            args.img_width // args.patch_size,
                            args.img_width // args.patch_size,
                            args.patch_size ** 2 * args.img_channel))
    #16 16 16 
    print("real_input_flag2 =",real_input_flag.shape)#(8, 9, 16, 16, 16)
#    print("real_input_flag[0] =",real_input_flag[0])
#    sys.exit()
  
    
    return eta, real_input_flag


def record_train(model, ims, real_input_flag, configs, itr, index ,num_of_batch_size, save_path, model_name):
    cost = model.train(ims, real_input_flag)#model_factory.py
    
    # if configs.reverse_input:
    #     ims_rev = np.flip(ims, axis=1).copy()
    #     print("ims_rev=",ims_rev.shape)
    #     cost += model.train(ims_rev, real_input_flag)
    #     cost = cost / 2

    if index == num_of_batch_size-1:
#        fn_path = save_path#!
        fn = model_name + '.txt'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fn = save_path + fn
        with open(fn,'a') as file_obj:
            file_obj.write( str(cost) + '\n')

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'itr: '+str(itr))
    print('training loss: ' + str(cost))
    del ims
    del real_input_flag
    return cost
#  {'Banqiao': 874, 'Keelung': 1749, 'Taipei': 2625, 'New_House': 3423, 'Chiayi': 4391,
#   'Dawu': 5346, 'Hengchun': 6308, 'Success': 7185, 'Sun_Moon_Lake': 8077, 'Taitung': 8951, 'Yuxi': 9802, 
#   'Hualien': 10779, 'Beidou': 11732, 'Bao_Zhong': 12697,
#   'Chaozhou': 13665, 'News': 14629, 'Member_Hill': 15534, 'Yuli': 16467, 'Snow_Ridge': 17338, 'Shangdewen': 18299}   
def batch_sample(train_generator, places, num_of_batch_size):
    sample_p20 = []
    sample_num={}
    for place in places:
        for index in range(num_of_batch_size):
            # batch_x, batch_y = train_generator.generator_getClassifiedItems_3(index, place)
            batch_x, batch_y = train_generator.generator_sample(index, place)
            batch_x = batch_x.astype(np.float16)  
            batch_y = batch_y.astype(np.float16)
            batch_y = batch_y.reshape(len(batch_y), 6, 64, 64, 1)
            if batch_y.shape[0] is not 0:
                bothx_y=np.concatenate((batch_x, batch_y), axis=1)
                sample_p20.append(bothx_y)
            else :
                continue
        print("place =",place,"bothx_y=",np.array(sample_p20).shape)
        sample_num[place] = len(sample_p20)
    
    print(sample_num)
    print("sample_p20 = ",np.array(sample_p20).shape)
    sample_p20 = np.array(sample_p20).reshape(-1,12,64,64,1)
    return sample_num
        # print("bothx_y.shape=",np.array(bothx_y).shape)   





def train_sample_wrapper(model, save_path, pretrained_model, model_name, train_radar_xy_shuffle,val_radar_xy_shuffle=None):    
    from pickle import load
    train_day_path = save_path+'train_day/'
    print("===train_sample_wrapper===")
    if pretrained_model is not None:
        print('pretrained_model',str(pretrained_model))
        model.load(save_path, pretrained_model)
    if train_radar_xy_shuffle is not None:
        places=['max places'] 
        print("all train_radar_xy_shuffle shape=",train_radar_xy_shuffle.shape)
        print("all val_radar_xy_shuffle shape=",val_radar_xy_shuffle.shape)

        num_of_batch_size = len(train_radar_xy_shuffle)//args.batch_size
        val_num_of_batch_size = len(val_radar_xy_shuffle)//args.batch_size

        print("train_radar_xy_shuffle num_of_batch_size=",num_of_batch_size)
        print("val num_of_batch_size=",val_num_of_batch_size)
    
    else:
        train_generator = radar.generator('train', batch_size=args.batch_size,save_path=train_day_path)
        num_of_batch_size = train_generator.step_per_epoch#!-1
    
    places=['Sun_Moon_Lake'] 
    place_len=len(places)
    # print("places=",places)
    # print("range(train_generator.step_per_epoch)=",range(train_generator.step_per_epoch))
    eta = args.sampling_start_value
    patience = 5
    min_loss = 100
    # patience = 2
    trigger_times = 0
    sample_p20 =[]
    print("place_len = ",place_len)
    for itr in range(5,500):
        # model.train()
        # print("model.training() = ",model.training())
        print("===========itr=",itr,"===========")
        # print("all train_radar_xy_shuffle shape=",train_radar_xy_shuffle.shape)
        sum_cost = 0
        avg_cost =0
        sum_xy_len=0
        sum_xy=np.zeros(0)
        smaple_p20_number = 0
        print("num_of_batch_size = ",num_of_batch_size)
        for place in places:
            smaple_number = 0
            cost_p1 = 0
            avg_p1_cost = 0
            print("place",place)#,"sample_num[place]",sample_num[place] )
            # num_of_batch_size = sample_num[place]
            for index in range(num_of_batch_size):
                batch_x, batch_y = train_generator.generator_getClassifiedItems_3(index, place)
                # batch_x, batch_y = train_generator.generator_getClassifiedItems_3(index, place)
                # batch_x, batch_y = train_generator.generator_sample(index, place)
                # train_xy = train_radar_xy_shuffle[index*args.batch_size:(index+1)*args.batch_size,:,:,:,:]#!
                print("num_of_batch_size * args.batch_size=",num_of_batch_size*args.batch_size,"index*args.batch_size=",index*args.batch_size,"to",(index+1)*args.batch_size)

                # batch_x,batch_y = np.split(train_xy, 2, axis=1)

                batch_x = batch_x.astype(np.float16)  
                batch_y = batch_y.astype(np.float16)
                
                # batch_x = np.array(batch_x).reshape(6,64*64)
                # scaler = StandardScaler()
                # batch_x=scaler.fit_transform(batch_x)
                # transformer = Normalizer().fit(batch_x)  # fit does nothing.
                # batch_x = transformer.transform(batch_x)
                # batch_x=np.array(batch_x).reshape(-1,6,64,64,1)


                scaler_path = save_path + 'srandard_scaler/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
                if not os.path.isdir(scaler_path):
                    os.makedirs(scaler_path)
                # scaler = load(open('min_max_scaler_8_240110.pkl', 'rb')) 
                # normalizer_scaler_8_240110
                '''
                scaler = load(open(scaler_path+'srandard_scaler_8_240210.pkl', 'rb')) 
                # scaler = load(open('normalizer_scaler_8_240210.pkl', 'rb'))   
                #    period=model_parameter['period'],
                #     predict_period=model_parameter['predict_period'],
                batch_x = np.array(batch_x).reshape(-1,64*64)
                batch_x = scaler.transform(batch_x)
                '''
                batch_x=np.array(batch_x).reshape(-1,model_parameter['period'],512,512,1)
                batch_y=np.array(batch_y).reshape(-1,model_parameter['predict_period'],512,512,1)

                ims=np.concatenate((batch_x, batch_y), axis=1)
                # ims = bothx_y

                ims = preprocess.reshape_patch(ims, args.patch_size)
#                print("np.array(ims==patch_tensor).shape=",np.array(ims).shape)
                #np.array(ims==patch_tensor).shape= (8, 20, 16, 16, 16)
#                [batch_size, seq_length,img_height//patch_size, patch_size,img_width//patch_size, patch_size,num_channels]
                eta, real_input_flag = schedule_sampling(eta, itr)
                print("type(ims)=",ims.dtype,"type(real_input_flag)=",real_input_flag.dtype)

#                trainer.train(model, ims, real_input_flag, args, itr, index ,num_of_batch_size, save_path, model_name)
                cost = record_train(model, ims, real_input_flag, args, itr, index ,num_of_batch_size, save_path, model_name)
                sum_xy=np.zeros(0)
                cost_p1 = cost_p1 + cost
                del batch_x
                del batch_y
                del real_input_flag
                del ims
            avg_p1_cost = cost_p1/num_of_batch_size#(smaple_number//32)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            
            fn = model_name + 'p20.txt'
            fn = save_path + fn
            with open(fn,'a') as file_obj:
                # file_obj.write('itr:' + str(itr) + 'place = '+str(place) +' training loss: ' + str(avg_p1_cost) + '\n')#應該不用除批次量
                file_obj.write(str(avg_p1_cost) + '\n')#應該不用除批次量
                
                # file_obj.write('num_of_batch_size*1:' + str(num_of_batch_size*1)+'\n') #!
                
                # file_obj.write('sample_num[place]:' + str(smaple_number)+'\n') 
                
            sum_cost = sum_cost + avg_p1_cost
            smaple_p20_number = smaple_p20_number + smaple_number

        avg_cost = sum_cost/place_len
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        fn = model_name + 'avg_p20.txt'
        fn = save_path + fn
        with open(fn,'a') as file_obj:
            file_obj.write('itr:' + str(itr) + 'place20,  training loss: ' + str(avg_cost) + '\n')#應該不用除批次量
            file_obj.write('smaple_p20_number:' + str(smaple_p20_number) + '\n')#應該不用除批次量

        model_pkl = model_name+'_itr{}.pkl'.format(itr)
        # if itr%100 == 0:
            # model.save(model_pkl,save_path) # val(model, save_path, model_pkl, itr)
        load_model=False
        test_cost, ssim = val(model, save_path, model_pkl, itr,val_radar_xy_shuffle=None)
        # test_cost = test_wrapper(model, save_path, model_pkl, itr,load_model=load_model)
        # if test_cost <= 0.001 or itr == 30000 or itr%100==0 :
        model_pkl = model_name+'_itr{}_test_cost{}_ssim{}.pkl'.format(itr, test_cost,ssim)
        model.save(model_pkl,save_path) # val(model, save_path, model_pkl, itr)
        if test_cost > min_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)
            
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                model_pkl = model_name+'_itr{}_earlystopping{}_test_cost{}_min_loss{}.pkl'.format(itr, patience, test_cost, min_loss)
                model.save(model_pkl,save_path)
                # return model
        else:
            trigger_times = 0
            print('trigger times: ',trigger_times)

            min_loss = test_cost
    
def csi_picture(img_out, test_ims, save_path,data_name='csi'):
        test_x_6 = test_ims
        img_out = img_out#t9~t18
        if not os.path.isdir(save_path):
            os.makedirs(save_path)       
        
        Color = ['#00FFFF', '#4169E1', '#0000CD', '#ADFF2F', '#008000', '#FFFF00', '#FFD700', '#FFA500', '#FF0000', '#8B0000', '#FF00FF', '#9932CC']

        ## CSI comput
        csi = []
        img_out_315 = img_out#!img_out[:,:,97:412,97:412,:]
        print("1 floa32 img_out_315.dtype=",img_out_315.dtype)  
        # img_out_315[img_out_315 <= 1] = 0

        test_x_6_315 = test_ims#!test_x_6[:,:,97:412,97:412,:]
        print("2 test_x_6_315=",np.array(test_x_6_315).shape)
        
        img_out_315 = img_out_315.astype(np.float16)
        test_x_6_315 = test_x_6_315.astype(np.float16)
        print("2 floa32 img_out_315.dtype=",img_out_315.dtype)  

        print("2 test_x_6_315=",np.array(test_x_6_315).shape)
        img_out_315_0 = np.array(img_out_315[0,0,:,:,:]).reshape(512,512)#!.reshape(315,315)
        test_x_6_315_0 =np.array(test_x_6_315[0,0,:,:,:]).reshape(512,512)#!.reshape(315,315)
        print("img_out_315_0=",np.array(img_out_315_0).shape)

        #! visualized_area_with_map(img_out_315_0, 'Sun_Moon_Lake', shape_size=[315,315], title='pred_to_010', savepath=save_path)
        #! visualized_area_with_map(test_x_6_315_0, 'Sun_Moon_Lake', shape_size=[315,315], title='test_to_010', savepath=save_path)
        visualized_area_with_map(img_out_315_0, 'Sun_Moon_Lake', shape_size=[512,512], title='pred_to_010', savepath=save_path)
        visualized_area_with_map(test_x_6_315_0, 'Sun_Moon_Lake', shape_size=[512,512], title='test_to_010', savepath=save_path)



        img_out_315_1 = np.array(img_out_315[0,1,:,:,:]).reshape(512,512)#!.reshape(315,315)
        test_x_6_315_1 = np.array(test_x_6_315[0,1,:,:,:]).reshape(512,512)#!.reshape(315,315)
        print("img_out_315_1=",np.array(img_out_315_1).shape)

        visualized_area_with_map(img_out_315_1, 'Sun_Moon_Lake', shape_size=[512,512], title='pred_to_020', savepath=save_path)
        visualized_area_with_map(test_x_6_315_1, 'Sun_Moon_Lake', shape_size=[512,512], title='test_to_020', savepath=save_path)

        for period in range(model_parameter['predict_period']):
            print("period=",period)
        #    print('pred_y[:, period] = ', pred_y[:, period])
        #    print('test_y[:, period] = ', test_y[:, period])
            csi_eva = Verification(pred=img_out_315[:, period].reshape(-1, 1), target=test_x_6_315[:, period].reshape(-1, 1), threshold=60, datetime='')
            print("csi_eva.csi.shape=",np.array(csi_eva.csi).shape)# (60, 99225)
            print("csi_eva.csi")
            print(csi_eva.csi)
            csi.append(np.nanmean(csi_eva.csi, axis=1))
            print("csi")

            print(csi)
            print("csi_eva.csi[0,:]")
            print(csi_eva.csi[0,:])
            print("mean",np.mean(csi_eva.csi[0,np.isfinite(csi_eva.csi[0,:])]))
            print("np.array(csi).shape=",np.array(csi).shape)#(1, 60)
            # sys.exit()
        
        csi = np.array(csi)
        np.savetxt(save_path+'{}.csv'.format(data_name), csi, delimiter = ',')
        # np.savetxt(save_path+'T202005270000csi.csv', csi.reshape(6,60), delimiter = ' ')

        ## Draw thesholds CSI
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        plt.xlim(0, 60)
        plt.ylim(-0.05, 1.0)
        plt.xlabel('Threshold')
        plt.ylabel('CSI')
        plt.title('{}\nThresholds CSI'.format(data_name))
        plt.grid(True)

        all_csi = []
        for period in range(model_parameter['predict_period']):
            plt.plot(np.arange(csi.shape[1]), [np.nan] + csi[period, 1:].tolist(), 'o--', label='{} min'.format((period+1)*10))

        plt.legend(loc='upper right')

        fig.savefig(fname=save_path+'Thresholds_CSI.png', format='png')
        plt.clf()


        ## Draw thesholds AVG CSI
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        plt.xlim(0, 60)
        plt.ylim(-0.05, 1.0)
        plt.xlabel('Threshold')
        plt.ylabel('CSI')
        plt.title('{}\nThresholds CSI'.format(data_name))
        plt.grid(True)

        all_csi = []
        plt.plot(np.arange(csi.shape[1]), [np.nan] + np.mean(csi[:, 1:], 0).tolist(), 'o--', label='AVG CSI')
        
        plt.legend(loc='upper right')

        fig.savefig(fname=save_path+'Thresholds_AVG_CSI.png', format='png')
        plt.clf()

        #csie = time.clock()
        #
        #alle = time.clock()
        #
        #print("load NWP time = ", loadNe - loadNs)
        #print("load CREF time = ", loadCe - loadCs)
        #print("All time = ", alle - alls)
        ## Draw peiod ALL CSI 
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        plt.xlim(0, model_parameter['predict_period']+1)
        plt.ylim(-0.05, 1.0)
        plt.xlabel('Time/10min')
        plt.ylabel('CSI')
        my_x_ticks = np.arange(0, model_parameter['predict_period']+1, 1)
        plt.xticks(my_x_ticks)
        plt.title('Threshold 5-55 dBZ')
        plt.grid(True)
        i = 0
        for threshold in range(5, 56, 5):
            plt.plot(np.arange(len(csi)+1), [np.nan] + list(csi[:, threshold-1]), 'o--', label='{} dBZ'.format(threshold), color=Color[i])
            i = i + 1
        #plt.legend(loc='lower right')

        plt.clf()

        fig.savefig(fname=save_path+'Period_CSI_ALL2.png', format='png')

        rmse_315=np.sqrt(((img_out_315 - test_x_6_315) ** 2).mean())

        rmse=np.sqrt(((img_out - test_x_6) ** 2).mean())

        fn = save_path + '{}_rmse.txt'.format(data_name)
        with open(fn,'a') as file_obj:
            file_obj.write('rmse=' + str(rmse)+'\n')

            file_obj.write('rmse_315=' + str(rmse_315)+'\n')
       

            # file_obj.write("-----test mse itr ="+str(itr)+"----- \n")
            # file_obj.write("model_pkl " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
            # file_obj.write("test day len(num_of_batch_size*1) =  " + str(num_of_batch_size*1)  + '\n' )
            # file_obj.write("place" + str(place)  + '\n' )
            
            # for i in range(args.total_length - args.input_length):
            #     print("sum 512x512 mse seq[",i,"] =",img_mse[i] / (batch_id * 1))#
            #     print("avg 512x512 mse seq[",i,"] =",((img_mse[i] / (batch_id * 1))/(512*512)))
            #     # file_obj.write("avg 512x512 mse seq[" + str(i) + '],test loss: ' + str((img_mse[i] /32)/(512*512)) + '\n')  
            #     file_obj.write("avg 512x512_2 mse seq[" + str(i) + '],test loss: ' + str(((img_mse[i] /1)/num_of_batch_size)/(512*512)) + '\n')  
            #     file_obj.write("avg 512x512_2 ssim seq[" + str(i) + '],test loss: ' + str(ssim[i]/batch_id * 1) + '\n')  

            # # file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
            # file_obj.write("test loss:" + str(avg_batch_cost) + '\n')     
            # file_obj.write("test mse:" + str(avg_mse_p1) + '\n')  
            # file_obj.write("test avg_ssim:" + str(avg_ssim) + '\n')  
def test_wrapper(model, save_path, model_pkl, itr,load_model=False ):
    print("===========test_wrapper===========")
    # model.eval()
    from pickle import load

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if load_model:
        model.load(save_path, model_name)
    place_len = len(places)
    main_path = save_path
    # save_path = save_path + 'test_wrapper_itr{}_20205M9D_2018313/'.format(itr)
    # save_path = save_path + 'test_wrapper_itr{}_201803130010to03132359/'.format(itr)
    # save_path = save_path + 'test_wrapper_itr{}_201808240010to08242359_v3/'.format(itr)
    # save_path = save_path + 'test_wrapper_itr{}_202005270010to05272359_v3/'.format(itr)
    # save_path = save_path + 'test_wrapper_itr{}_202005270010to20_v3/'.format(itr)
    save_path = save_path + 'test_wrapper_itr{}_201808240010to20_v4/'.format(itr)
    # 
    # save_path = save_path + 'test_wrapper_itr{}_201808240010_v3/'.format(itr)
    # save_path = save_path + 'test_wrapper_itr{}_201803130010_v3/'.format(itr)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # model.load_state_dict(torch.load('params.pkl')) 
    # places=['Bao_Zhong']
    # date_date=[['2018-03-13 00:10', '2018-03-13 23:59']]
    # test_date=[['2018-03-13 00:10', '2018-03-13 23:59']]
    # date_date=[['2018-03-13 00:10', '2018-03-13 00:19']]
    # test_date=[['2018-03-13 00:10', '2018-03-13 00:19']]

    # date_date=[['2018-08-24 00:10', '2018-08-24 23:59']]
    # test_date=[['2018-08-24 00:10', '2018-08-24 23:59']]
    date_date=[['2018-08-24 00:10', '2018-08-24 00:29']]
    test_date=[['2018-08-24 00:10', '2018-08-24 00:29']]

    # date_date=[['2020-05-27 00:10', '2020-05-27 23:59']]
    # test_date=[['2020-05-27 00:10', '2020-05-27 23:59']]
    # date_date=[['2020-05-27 00:10', '2020-05-27 00:29']]
    # test_date=[['2020-05-27 00:10', '2020-05-27 00:29']]

    # places=['Banqiao','Keelung','Taipei','New_House','Chiayi',
    #     'Dawu','Hengchun','Success','Sun_Moon_Lake','Taitung',#,
    #     'Yuxi','Hualien','Beidou','Bao_Zhong','Chaozhou',
    #     'News','Member_Hill','Yuli','Snow_Ridge','Shangdewen']
    #        5/16、19、21、22、23、26、27、28、29
    # # places=['Banqiao','Keelung']
    '''
    date_date=[['2018-03-13 00:00','2018-08-13 23:59'],
        ['2020-05-16 00:00','2020-05-16 23:59'],
        ['2020-05-19 00:00','2020-05-19 23:59'],
        ['2020-05-21 00:00','2020-05-21 23:59'],
        ['2020-05-22 00:00','2020-05-22 23:59'],
        ['2020-05-23 00:00','2020-05-23 23:59'],
        ['2020-05-26 00:00','2020-05-26 23:59'],
        ['2020-05-27 00:00','2020-05-27 23:59'],
        ['2020-05-28 00:00','2020-05-28 23:59'],
        ['2020-05-29 00:00','2020-05-29 23:59']]
    val_data=[['2018-03-13 00:00','2018-08-13 23:59'],
    ['2020-05-16 00:00','2020-05-16 23:59'],
            ['2020-05-19 00:00','2020-05-19 23:59'],
            ['2020-05-21 00:00','2020-05-21 23:59'],
            ['2020-05-22 00:00','2020-05-22 23:59'],
            ['2020-05-23 00:00','2020-05-23 23:59'],
            ['2020-05-26 00:00','2020-05-26 23:59'],
            ['2020-05-27 00:00','2020-05-27 23:59'],
            ['2020-05-28 00:00','2020-05-28 23:59'],
            ['2020-05-29 00:00','2020-05-29 23:59']]
    '''
    radar_echo_storage_path= 'D:/yu_ting/try/NWP/'#'NWP/'
    load_radar_echo_df_path=main_path+'2018_7mto8m_Sun_Moon_Lake_512x512_T12toT6.pkl'
    radar_test = load_data(radar_echo_storage_path=radar_echo_storage_path, 
                    load_radar_echo_df_path=load_radar_echo_df_path,#'data/RadarEcho_Bao_Zhong_2018_08240010_T6toT6_inoutputshape64_random.pkl',#None,#load_radar_echo_df_path,
                    input_shape=[512,512],#[512,512],#model_parameter['input_shape'],
                    output_shape=[512,512],#model_parameter['output_shape'],
                    period=model_parameter['period'],
                    predict_period=model_parameter['predict_period'],
                    places=places,
                    random=False,
                    date_range=date_date,
                    test_date=test_date)
    if not load_radar_echo_df_path:
        radar_test.exportRadarEchoFileList()
    #     radar_tw.saveRadarEchoDataFrame()
        radar_test.saveRadarEchoDataFrame(path=save_path ,load_name_pkl='2020_9day_20180313_Sun_Moon_Lake_512x512')   

    
    test_day_path = save_path+'test_day/'

    test_generator = radar_test.generator('test', batch_size=1)#)args.batch_size,save_path = test_day_path )
   
    batch_id = 0
    img_mse, ssim = [], []

    real_input_flag = np.zeros(
        (1,#args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    p20_cost =0
    avg_p20_cost=0

    p20_mse =0
    for place in places: 
        print("places=",places)
        print("range(test_generator.step_per_epoch)=",range(test_generator.step_per_epoch))
        num_of_batch_size = test_generator.step_per_epoch#!-1
        batch_id = 0
        batch_cost = 0
        avg_batch_cost = 0
        
        sum_mse_index = 0
        img_mse, ssim = [], []
        avg_ssim=0
        for i in range(args.total_length - args.input_length):
            img_mse.append(0)
            ssim.append(0)
        for index in range(num_of_batch_size):
            batch_id = batch_id + 1
            batch_x, batch_y = test_generator.generator_getClassifiedItems_3(index, place)
            batch_x = batch_x.astype(np.float16)  
            batch_y = batch_y.astype(np.float16)
            batch_y = batch_y.reshape(len(batch_y), model_parameter['predict_period'], 512, 512, 1)
            scaler_path = main_path + 'srandard_scaler/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
            if not os.path.isdir(scaler_path):
                os.makedirs(scaler_path)
            # scaler = load(open('min_max_scaler_8_240210.pkl', 'rb'))       
            # srandard
            # scaler = load(open(scaler_path+'srandard_scaler_8_240210.pkl', 'rb'))       
            # scaler = load(open('normalizer_scaler_8_240210.pkl', 'rb'))   
            # batch_x = np.array(batch_x).reshape(6,512*512)
            # batch_x = scaler.transform(batch_x)
            batch_x=np.array(batch_x).reshape(-1,model_parameter['period'],512,512,1)


            bothx_y=np.concatenate((batch_x, batch_y), axis=1)
            val_ims = np.stack(bothx_y,axis=0)   
            print("np.array(val_ims).shape=",np.array(val_ims).shape)
            val_dat = preprocess.reshape_patch(val_ims, args.patch_size)
            print("test_dat  preprocess.reshape_patch=",np.array(val_dat).shape)
            img_gen = model.test(val_dat, real_input_flag)
            print("--預測-")
            print("img_gen model.test=",np.array(img_gen).shape)
            img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
            print("img_gen preprocess.reshape_patch_back=",np.array(img_gen).shape)
            output_length = args.total_length - args.input_length
            # img_gen_length = img_gen.shape[1]
            img_out = img_gen[:, -output_length:]
            # img_out_all.append(img_out)
            test_x_6 = val_ims[:, model_parameter['period']:, :, :, :]
            # test_x_6_all.append(test_x_6)
            if batch_id ==1:
                test_ims_all = test_x_6
                img_out_all = img_out
            else:
                test_ims_all = np.concatenate((test_ims_all , test_x_6) ,axis = 0)#old nee
                img_out_all = np.concatenate((img_out_all , img_out) ,axis = 0)#old nee
            
            print("test_ims_all",np.array(test_ims_all).shape)
            print("img_out_all",np.array(img_out_all).shape)

            mse = np.square(test_x_6 - img_out).sum()
            mse_picture_avg = ((mse/1)/model_parameter['predict_period'])/(512*512)
            sum_mse_index = sum_mse_index + mse_picture_avg
            # MSE per frame
            save_path_single_location = save_path+'512x512_test_9d/'
            seq_p1_cost =0
            sum_mse =0
            avg_seq_p1_cost= 0
            for i in range(output_length):
                x = val_ims[:, i + args.input_length, :, :, :]
                gx = img_out[:, i, :, :, :]
                # gx = np.maximum(gx, 0)
                # gx = np.minimum(gx, 60)
                print("x.shape=",x.shape,"gx=",gx.shape)#x.shape= (8, 512, 512, 1) gx= (8, 512, 512, 1)
                mse = np.square(x - gx).sum()
                img_mse[i] += mse
                # avg_mse += mse
                sum_mse +=mse
                account_mse = sum_mse/1

                # account_mse = mse/1
                seq_p1_cost = seq_p1_cost + account_mse 
                print("batch_id=",batch_id,"i + args.input_length=",i + args.input_length,"account_mse=",account_mse)
                vis_x = x.copy()
                # vis_x[vis_x > 1.] = 1.
                # vis_x[vis_x < 0.] = 0.
                vis_gx = gx.copy()
                vis_x = vis_x/65
                vis_gx = vis_gx/65
                # vis_gx[vis_gx > 1.] = 1.
                # vis_gx[vis_gx < 0.] = 0.

                real_frm = np.uint8(vis_x * 255).reshape(512,512)
                pred_frm = np.uint8(vis_gx * 255).reshape(512,512)

                # for b in range(configs.batch_size):
                score, _ = compare_ssim(pred_frm, real_frm,data_range=255, full=True, multichannel=True)
                ssim[i] += score     
                avg_ssim+=score
                del x
                del gx
            avg_seq_p1_cost = ((sum_mse/1)/model_parameter['predict_period'])/(512*512)

            batch_cost = batch_cost + avg_seq_p1_cost
            del batch_x
            del batch_y
            del real_input_flag
            # del ims
            del val_ims
            del img_gen
            del img_out
            del val_dat

            del test_x_6
            del test_ims_all
            del img_out_all
        avg_batch_cost =batch_cost/num_of_batch_size
        avg_mse_p1 = sum_mse_index/num_of_batch_size
        # print('mse per seq: ' + str(avg_mse))    
        print("ssim",np.array(ssim).shape)
        avg_ssim = (avg_ssim/model_parameter['predict_period'])/num_of_batch_size
        save_path_index = save_path_single_location + '512x512_PD/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
        if not os.path.isdir(save_path_index):
            os.makedirs(save_path_index)
        fn = save_path_index + 'test_mse_div20_.txt'
        with open(fn,'a') as file_obj:
            file_obj.write("-----test mse itr ="+str(itr)+"----- \n")
            file_obj.write("model_pkl " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
            file_obj.write("test day len(num_of_batch_size*1) =  " + str(num_of_batch_size*1)  + '\n' )
            file_obj.write("place" + str(place)  + '\n' )
            
            for i in range(args.total_length - args.input_length):
                print("sum 512x512 mse seq[",i,"] =",img_mse[i] / (batch_id * 1))#
                print("avg 512x512 mse seq[",i,"] =",((img_mse[i] / (batch_id * 1))/(512*512)))
                # file_obj.write("avg 512x512 mse seq[" + str(i) + '],test loss: ' + str((img_mse[i] /32)/(512*512)) + '\n')  
                file_obj.write("avg 512x512_2 mse seq[" + str(i) + '],test loss: ' + str(((img_mse[i] /1)/num_of_batch_size)/(512*512)) + '\n')  
                file_obj.write("avg 512x512_2 ssim seq[" + str(i) + '],test loss: ' + str(ssim[i]/batch_id * 1) + '\n')  

            # file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
            file_obj.write("test loss:" + str(avg_batch_cost) + '\n')     
            file_obj.write("test mse:" + str(avg_mse_p1) + '\n')  
            file_obj.write("test avg_ssim:" + str(avg_ssim) + '\n')  


        p20_mse = p20_mse+ avg_mse_p1
        p20_cost = p20_cost + avg_batch_cost
    p20_mse = p20_mse/place_len
    avg_p20_cost = p20_cost/place_len
    # avg_ssim = avg_ssim/place_len#!
    fn = save_path_single_location + 'test_mse_avg20_.txt'
    with open(fn,'a') as file_obj:
        file_obj.write("-----test mse itr ="+str(itr)+"----- \n")
        file_obj.write("model_pkl " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
        file_obj.write("test day len(num_of_batch_size*1) =  " + str(num_of_batch_size*1)  + '\n' )
        file_obj.write("place 20"   + '\n' )
        file_obj.write("test avg_ssim:" + str(avg_ssim) + '\n')  

        
        # for i in range(args.total_length - args.input_length):
        #     print("sum 512x512 mse seq[",i,"] =",img_mse[i] / (batch_id * args.batch_size))#
        #     print("avg 512x512 mse seq[",i,"] =",((img_mse[i] / (batch_id * args.batch_size))/(512*512))/4)
        #   file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
        file_obj.write("test loss:" + str(avg_p20_cost) + '\n') 

        file_obj.write("p20_mse mse:" + str(p20_mse) + '\n')  



    fn = save_path_single_location + 'test_mse_itr.txt'
    with open(fn,'a') as file_obj:
        # file_obj.write("-----test mse itr ="+str(itr)+"----- \n")
        # file_obj.write("model_pkl " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
        # file_obj.write("test day len(num_of_batch_size*1) =  " + str(num_of_batch_size*args.batch_size)  + '\n' )
        # file_obj.write("place 20"   + '\n' )
        
        # for i in range(args.total_length - args.input_length):
        #     print("sum 512x512 mse seq[",i,"] =",img_mse[i] / (batch_id * args.batch_size))#
        #     print("avg 512x512 mse seq[",i,"] =",((img_mse[i] / (batch_id * args.batch_size))/(64*64))/4)
        #   file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
        file_obj.write(str(p20_mse) + '\n') 

        # file_obj.write("p20_mse mse:" + str(p20_mse) + '\n')  
  
    csi_picture(img_out = img_out_all,test_ims= test_ims_all,save_path = save_path+'csi/')
    return p20_mse

def test_show(model, save_path, model_name,itr):
    print("===========test_show===========")

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    model.load(save_path, model_name)
    main_path = save_path
    # save_path = save_path + 'test_show_itr_{}_201808240010_Sun_Moon_Lake_dbz1_315X315/'.format(itr)
    # save_path = save_path + 'test_show_itr_{}_202005190010_Sun_Moon_Lake_dbz1/'.format(itr)

    # save_path = save_path + 'test_show_itr_{}_202005270010to5272359_Sun_Moon_Lake_dbz1_315X315/'.format(itr)
    # data_name='201905170010'
    # data_name='202005270010'
    # data_name='202005160800to6'
    # data_name='202005280000to6'
    data_name='202005270000to6'
    # data_name='201704211100to6'
    
    # data_name='201808240000to6'
    # data_name='201405200500to6'
    # data_name='202008260400to6'

    # data_name='202005220400to6'
    # data_name='202005290600to6'
    # date_date=[['2020-05-19 00:10', '2020-05-19 00:19']]
    # data_name='202005190000to6'
    
    

    save_path = save_path + 'test_show_itr_{}_{}_Sun_Moon_Lake_dbz1_csi_testcsi/'.format(itr,data_name)
    from pickle import load
    # places=['Bao_Zhong']
    # date_date=[['2018-08-23 22:10', '2018-08-24 03:59']]
    # test_date=[['2018-08-24 00:10', '2018-08-24 01:00']]
    # test_date=[['2018-03-13 00:10', '2018-03-13 23:59']]
    # date_date=[['2020-05-20 22:10', '2020-05-21 02:59']]
    # test_date=[['2020-05-21 00:10', '2020-05-21 00:11']]
    # date_date=[['2018-08-23 00:00', '2018-08-23 23:59']]
    # test_date=[['2018-08-23 01:10', '2018-08-23 01:11']]
    places=['Sun_Moon_Lake']
    place_len=len(places)
    # date_date=[['2018-06-15 12:00', '2018-06-15 12:01']]
    # test_date=[['2018-06-15 12:00', '2018-06-15 12:01']]
    # date_date=[['2018-08-24 00:00', '2018-08-24 23:59']]

    # test_date=[['2018-08-24 00:10', '2018-08-24 00:11']]
    # date_date=[['2018-08-24 00:10', '2018-08-24 00:11']]
    
    # test_date=[['2020-08-26 04:10', '2020-08-26 04:11']]
    # date_date=[['2020-08-26 04:10', '2020-08-26 04:11']]
    # test_date=[['2014-05-20 05:10', '2014-05-20 05:11']]
    # date_date=[['2014-05-20 05:10', '2014-05-20 05:11']]

    # test_date=[['2020-05-22 04:10', '2020-05-22 04:11']]
    # date_date=[['2020-05-22 04:10', '2020-05-22 04:11']]

    # test_date=[['2020-05-29 06:10', '2020-05-29 06:11']]
    # date_date=[['2020-05-29 06:10', '2020-05-29 06:11']]

    # test_date=[['2018-08-24 03:10', '2018-08-24 03:11']]
    # date_date=[['2018-05-09 00:00', '2018-05-09 00:01']]
    # test_date=[['2018-05-09 00:00', '2018-05-09 00:01']]


    date_date=[['2020-05-27 00:10', '2020-05-27 00:19']]
    test_date=[['2020-05-27 00:10', '2020-05-27 00:19']]
    # date_date=[['2017-04-21 11:10', '2017-04-21 11:19']]
    # test_date=[['2017-04-21 11:10', '2017-04-21 11:19']]

# 2017-04-21 10:00
    # date_date=[['2020-08-26 00:10', '2020-08-26 00:19']]
    # test_date=[['2020-08-26 00:10', '2020-08-26 00:19']]

    # date_date=[['2019-05-17 00:10', '2019-05-17 00:19']]
    # test_date=[['2019-05-17 00:10', '2019-05-17 00:19']]


    # date_date=[['2020-05-19 00:10', '2020-05-19 00:19']]
    # test_date=[['2020-05-19 00:10', '2020-05-19 00:19']]

    # date_date=[['2020-05-16 08:10', '2020-05-16 08:11']]
    # test_date=[['2020-05-16 08:10', '2020-05-16 08:11']]


    # date_date=[['2020-05-28 00:10', '2020-05-28 00:11']]
    # test_date=[['2020-05-28 00:10', '2020-05-28 00:11']]

    # date_date=[['2018-08-23 00:10', '2018-08-23 00:10']]
    # test_date=[['2018-08-23 00:10', '2018-08-23 00:10']]
    load_radar_echo_df_path='InterDST_LSTM_L2_H64_2017to2019_pickday_checkpoint_pretrain_loss_v2/202005270000to6_512x512.pkl'
    
    radar_echo_storage_path= 'E:/yu_ting/try/NWP/'#'NWP/'
    # load_radar_echo_df_path='data/RadarEcho_p20_2018_2019_6mto8m_T6toT6_inoutputshape64_random.pkl'
    # load_radar_echo_df_path="T18toT6_['Sun_Moon_Lake']_512x512_v2/save_2018_801to830/2018_m8_Sun_Moon_Lake_512x512_T18toT6.pkl"#None#'samping_test/save_2019_2018_2019_3m_p20_sample6/2018_2018_2019_3m_p20_inputsize64_random.pkl'#None#'sample_p20/save_2018_3m_p4_sample/2018_3m_p4_random.pkl'#None#'sample/save_2018_3m_sample/2018_20day.pkl'#None#'data/RadarEcho_p20_2018_2019_6mto8m_T6toT6_inoutputshape64_random.pkl'
    # D:\yu_ting\predrnn\predrnn_gogo\T10toT10_['Sun_Moon_Lake']_512x512\save_2018_801to830
    radar_1p = load_data(radar_echo_storage_path=radar_echo_storage_path, 
                    load_radar_echo_df_path=load_radar_echo_df_path,#'data/RadarEcho_Bao_Zhong_2018_08240010_T6toT6_inoutputshape64_random.pkl',#None,#load_radar_echo_df_path,
                    input_shape=[512,512],#[512,512],#model_parameter['input_shape'],
                    output_shape=[512,512],#model_parameter['output_shape'],
                    period=model_parameter['period'],
                    predict_period=model_parameter['predict_period'],
                    places=places,
                    random=False,
                    date_range=date_date,
                    test_date=test_date,
                    save_np_radar=save_path )
    if not load_radar_echo_df_path:
        radar_1p.exportRadarEchoFileList()
    #     radar_tw.saveRadarEchoDataFrame()
        radar_1p.saveRadarEchoDataFrame(path=save_path ,load_name_pkl='{}_512x512'.format(data_name))   

    


    test_show_day_path = save_path+'teat_show_day/'
    test_generator = radar_1p.generator('test', batch_size=1,save_path = test_show_day_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim = [], []
    test_x_6 =[]
    img_out=[]
    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (1,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    num_of_batch_size = test_generator.step_per_epoch
    avg_ssim=0
    test_ims_all=[]
    img_out_all=[]
    sum_p20_mse_picture_avg=0
    for place in places: 
        print("places=",places)
        print("range(test_generator.step_per_epoch)=",range(test_generator.step_per_epoch))
        sum_p1_mse_picture_avg=0
        sum_mse_index =0
        batch_cost=0
        batch_id=0
        for index in range(num_of_batch_size):
            batch_id = batch_id + 1
            batch_x, batch_y = test_generator.generator_getClassifiedItems_3(index, place)
            # batch_x, batch_y = test_generator.generator_sample(index, place)  

            # batch_x = np.array(batch_x).reshape(6,64*64)

            # batch_x = scaler.transform(batch_x)
            '''
            transformer = Normalizer().fit(batch_x)
            batch_x = transformer.transform(batch_x)

            batch_x=np.array(batch_x).reshape(-1,6,64,64,1)
            '''
            
            # scaler = load(open('min_max_scaler_8_240210.pkl', 'rb'))       
            # srandard
            scaler_path = main_path + 'srandard_scaler/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
            '''
            if not os.path.isdir(scaler_path):
                os.makedirs(scaler_path)
            scaler = load(open(scaler_path+'srandard_scaler_8_240210.pkl', 'rb'))       
            # scaler = load(open('normalizer_scaler_8_240210.pkl', 'rb'))   
            '''
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            np.savetxt(save_path+'radar_8_240210_testx.txt', batch_x.reshape(-1), delimiter=' ')
            np.savetxt(save_path+'radar_8_240210_testy.txt', batch_y.reshape(-1), delimiter=' ')
            '''
            batch_x = np.array(batch_x).reshape(model_parameter['period'],64*64)
            batch_x = scaler.transform(batch_x)
            '''
            batch_x=np.array(batch_x).reshape(-1,model_parameter['period'],512,512,1)

            batch_x = batch_x.astype(np.float16)  
            batch_y = batch_y.astype(np.float16)
            batch_y = batch_y.reshape(len(batch_y), model_parameter['predict_period'], 512, 512, 1)
            
            bothx_y=np.concatenate((batch_x, batch_y), axis=1)
            test_ims = np.stack(bothx_y,axis=0)   
            print("np.array(test_ims).shape=",np.array(test_ims).shape)

            test_dat = preprocess.reshape_patch(test_ims, args.patch_size)
            print("test_dat  preprocess.reshape_patch=",np.array(test_dat).shape)
            img_gen = model.test(test_dat, real_input_flag)
            print("--預測-")
            print("img_gen model.test=",np.array(img_gen).shape)
            img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
            print("img_gen preprocess.reshape_patch_back=",np.array(img_gen).shape)
            output_length = args.total_length - args.input_length
            img_gen_length = img_gen.shape[1]
            img_out = img_gen[:, -output_length:]

            np.savetxt(save_path+'radar_8_240210_img_outy.txt', batch_y.reshape(-1), delimiter=' ')

            test_x_6 = test_ims[:,model_parameter['period']:,:,:,:]
            test_mse = np.square(test_x_6 - img_out).sum()



            if batch_id ==1:
                test_ims_all = test_x_6
                img_out_all = img_out
            else:
                test_ims_all = np.concatenate((test_ims_all , test_x_6) ,axis = 0)#old nee
                img_out_all = np.concatenate((img_out_all , img_out) ,axis = 0)#old nee
            
            print("test_ims_all",np.array(test_ims_all).shape)
            print("img_out_all",np.array(img_out_all).shape) 


            mse_picture_avg = ((test_mse/1)/model_parameter['predict_period'])/(512*512)
            sum_mse_index = sum_mse_index + mse_picture_avg
            # MSE per frame
            save_path_single_location = save_path+'512x512_test_tw/'
            save_path_index = save_path_single_location + '512x512_PD_index_{}_test/'.format(index)#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
            # for one_size in range(8):
            sum_mse =0
            avg_seq_p1_cost=0
            mae_img=0
            for i in range(output_length):
                x = test_ims[:, i + args.input_length, :, :, :]
                gx = img_out[:, i, :, :, :]
                # gx = np.maximum(gx, 1)
                # gx = np.minimum(gx, 6)

                # gx = np.minimum(gx, 60)
                print("x.shape=",x.shape,"gx=",gx.shape)#x.shape= (8, 512, 512, 1) gx= (8, 512, 512, 1)
                mse = np.square(x - gx).sum()
                img_mse[i] += mse
                avg_mse += mse
                sum_mse +=mse
                account_mse = sum_mse/1
                # seq_p1_cost = seq_p1_cost + account_mse

            avg_seq_p1_cost = ((sum_mse/1)/model_parameter['predict_period'])/(512*512)
            batch_cost = batch_cost + avg_seq_p1_cost

            for one_size in range(1):
                save_test_path = save_path_index + 'batch_{}/'.format(one_size)
                if not os.path.isdir(save_test_path):
                    os.makedirs(save_test_path)
                mae_img=0
                for i in range(output_length):
                    # print("x.shape=",x.shape,"gx=",gx.shape)#x.shape= (8, 512, 512, 1) gx= (8, 512, 512, 1)
                    vis_gx = np.array(img_out[one_size, i, :, :, :]).reshape(512,512)#t9~t18
                    vis_x = np.array(test_ims[one_size,  i + args.input_length, :, :, :]).reshape(512,512)#10,11,12,13,14,15,16,17,18,19
                    vis_test_x = np.array(test_ims[one_size,  i , :, :, :]).reshape(512,512)#10,11,12,13,14,15,16,17,18,19
                    
                    # vis_test_x = scaler.inverse_transform(vis_test_x.reshape(-1))
                    vis_test_x = vis_test_x.reshape(512,512)
                    # print("vis_gx.shape = ",vis_gx.shape)
                    # vis_gx = np.maximum(vis_gx, 1)
                    # vis_gx = vis_gx+3
                    vis_gx[vis_gx <= 1] = 0
                    # vis_gx = np.minimum(vis_gx, 70)                
                    visualized_area_with_map(vis_gx, 'Sun_Moon_Lake', shape_size=[512,512], title='vis_pred_{}'.format(i), savepath=save_test_path)
                    # visualized_area_with_map(vis_x, 'Sun_Moon_Lake', shape_size=[512,512], title='vis_gt_{}'.format(i), savepath=save_test_path)
                    # visualized_area_with_map(vis_test_x, 'Sun_Moon_Lake', shape_size=[512,512], title='vis_test_x_{}'.format(i), savepath=save_test_path)
                    # mae_img  = vis_x-vis_gx
                    # print("mae_img.shape=",mae_img.shape)
                    # mae_img=mae_img.reshape(512,512)
                    # visualized_area_with_map_mae(mae_img, 'Sun_Moon_Lake', shape_size=[512,512], title='vis_x-vis_gx_{}'.format(i), savepath=save_test_path)
                    # mae_img  = vis_gx-vis_x
                    # mae_img=mae_img.reshape(512,512)
                    # visualized_area_with_map_mae(mae_img, 'Sun_Moon_Lake', shape_size=[512,512], title='vis_gx-vis_x_{}'.format(i), savepath=save_test_path)
                    
                    
                    fn = save_test_path+'_sqe{}.txt'.format(i)
                    mse = np.square(vis_x - vis_gx).sum()
                    div_h_w = mse/(512*512)
                    # vis_x[vis_x > 1.] = 1.
                    # vis_x[vis_x < 0.] = 0.

                    # vis_gx[vis_gx > 1.] = 1.
                    # vis_gx[vis_gx < 0.] = 0.
                    # vis_x = np.maximum(vis_x, 0)
                    # vis_x = np.minimum(vis_x, 1)

                    # vis_gx = np.maximum(vis_gx, 0)
                    # vis_gx = np.minimum(vis_gx, 1)
                    vis_x = vis_x/65
                    vis_gx = vis_gx/65

                    real_frm = np.uint8(vis_x * 255)
                    pred_frm = np.uint8(vis_gx * 255)

                    # real_frm =vis_x 
                    # pred_frm = vis_gx 
                    # for b in range(configs.batch_size):
                    # score, _ = compare_ssim(pred_frm, real_frm, full=True, multichannel=True)
                    # ssim[i] += score
                    # avg_ssim+=score
                    
                    with open(fn,'a') as file_obj:
                        file_obj.write("model_name " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
                    # file_obj.write("test day len(test_generator.step_per_epoch*32) =  " + str(test_generator.step_per_epoch*1)  + '\n' )
                        file_obj.write("test batch "+str(one_size) +" i = " + str(i) +"mse = "+str(div_h_w)   + '\n' )
                        # file_obj.write("test batch "+str(one_size) +" i = " + str(i) +"SSIM = "+str(score)   + '\n' )
        avg_mse_p1 = sum_mse_index/num_of_batch_size
        avg_batch_cost =batch_cost/num_of_batch_size
        # avg_ssim = (avg_ssim/model_parameter['predict_period'])/num_of_batch_size
        # avg_ssim = np.mean(ssim)
        print("mse=",img_mse)
        # avg_mse = avg_mse / (batch_id * 1)
        print('mse per seq: ' + str(avg_mse))
        
        fn = save_path_single_location + 'test_mse.txt'
        with open(fn,'a') as file_obj:
            file_obj.write("model_name " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
            file_obj.write("test day len(test_generator.step_per_epoch*32) =  " + str(test_generator.step_per_epoch*1)  + '\n' )
            file_obj.write("test day num_of_batch_size =  " + str(num_of_batch_size)  + '\n' )
           
            for i in range(args.total_length - args.input_length):
                # print("sum 512x512 mse seq[",i,"] =",img_mse[i] / (batch_id * 1))#
                print("avg 512x512 mse seq[",i,"] =",(img_mse[i] / (batch_id * 1))/(512*512))
                print("avg 512x512 ssim seq[",i,"] =",(ssim[i]))

    #           file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
                file_obj.write("avg 512x512 mse seq[" + str(i) + '],test loss: ' + str((img_mse[i] / (batch_id * 1))/(512*512)) + '\n')  
                # file_obj.write("avg 512x512 ssim seq[" + str(i) + '],test loss: ' + str(ssim[i]) + '\n')  
                # file_obj.write("avg 512x512_2 ssim seq[" + str(i) + '],test loss: ' + str(ssim[i]/batch_id * 1) + '\n')  
                
            file_obj.write("test avg_mse =  " + str(avg_batch_cost)  + '\n' )
            file_obj.write("test avg_mse_p1 =  " + str(avg_mse_p1)  + '\n' )
            # file_obj.write("test avg_ssim =  " + str(avg_ssim)  + '\n' )

        test_y_csi = test_ims[:,model_parameter['period']:,:,:,:]
        vis_gx_csi = img_out#t9~t18
    csi_picture(img_out = img_out_all,test_ims= test_ims_all,save_path = save_path+'csi_{}/'.format(data_name),data_name=data_name)
def val(model, save_path, model_pkl, itr, val_radar_xy_shuffle = None): 
    print("===========val_wrapper===========")
    # model.eval()
    # print("model.training() = ",model.training())
    from pickle import load

    main_path=save_path
    val_day=save_path+'val_day/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # model.load(save_path, model_pkl)
    # model.load_state_dict(torch.load('params.pkl'))
    if val_radar_xy_shuffle is not None:
        places=['max places'] 
        print("all val_radar_xy_shuffle shape=",val_radar_xy_shuffle.shape)
        num_of_batch_size = len(val_radar_xy_shuffle)//args.batch_size
        print("val_radar_xy_shuffle num_of_batch_size=",num_of_batch_size)
    else:
        val_generator = radar.generator('val', batch_size=1, save_path = val_day)#args.batch_size)
        num_of_batch_size = val_generator.step_per_epoch#!-1
        print("range(val_generator.step_per_epoch)=",range(val_generator.step_per_epoch))
        places=['Sun_Moon_Lake'] 
        
    avg_mse = 0
    batch_id = 0
    img_mse, ssim = [], []
    place_len=len(places)

    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    p20_cost =0
    avg_p20_cost=0
    sum_mse = 0
    p20_mse =0
    sum_xy=np.zeros(0)

    for place in places: 
        print("places=",places)
        # num_of_batch_size = val_generator.step_per_epoch-1
        batch_id = 0
        batch_cost = 0
        avg_batch_cost = 0
        sum_mse = 0
        sum_mse_index = 0
        smaple_p20_number = 0
        smaple_number=0
        avg_ssim = 0
        sum_ssim=0
        for index in range(num_of_batch_size):
            
            batch_id = batch_id + 1
            batch_x, batch_y = val_generator.generator_getClassifiedItems_3(index, place)
            batch_x = batch_x.astype(np.float16)  
            batch_y = batch_y.astype(np.float16)
            batch_y = batch_y.reshape(len(batch_y), model_parameter['predict_period'], 512, 512, 1)
            # bothx_y=np.concatenate((batch_x, batch_y), axis=1)
            # val_ims = np.stack(bothx_y,axis=0)   
            # print("np.array(val_ims).shape=",np.array(val_ims).shape)
            
            if val_radar_xy_shuffle is not None:
                train_xy = val_radar_xy_shuffle[index*args.batch_size:(index+1)*args.batch_size,:,:,:,:]#!
                print("index*args.batch_size=",index*args.batch_size,"to",(index+1)*args.batch_size)
                batch_x,batch_y = np.split(train_xy, 2, axis=1)
                print("batch_x=",batch_x.shape,"batch_y",batch_y.shape)
            # else:
            #     batch_x, batch_y = val_generator.generator_sample(index, place)#!
            # batch_x = batch_x.astype(np.float16)  
            # batch_y = batch_y.astype(np.float16)
            # batch_y = batch_y.reshape(len(batch_y), model_parameter['predict_period'], 512, 512, 1)



            # '''
            # if batch_y.shape[0] is 0:
            #     print("batch_x is zero(<5).shape=",batch_x.shape)#(1, 6, 64, 64, 1)
            #     continue
            # else:
            #     print("batch_x.shape=",batch_x.shape)#(1, 6, 64, 64, 1)
            #     print("batch_y.shape=",batch_y.shape)#(1, 6, 64, 64, 1)
            #     bothx_y_temp = np.concatenate((batch_x, batch_y), axis=1)
            #     smaple_number+=1
            # if sum_xy.shape[0] is 0:#第一次
            #     print("creat sum_xy before =",sum_xy.shape)
            #     sum_xy = bothx_y_temp
            #     print("creat sum_xy after =",sum_xy.shape)
            #     continue

            # if sum_xy.shape[0] < 4 and sum_xy is not False:#第二次到31次
            #     # print("add sum_xy before =",sum_xy.shape)(31, 12, 64, 64, 1)
            #     sum_xy = np.concatenate((bothx_y_temp, sum_xy), axis=0)
            #     # print("add sum_xy after =",sum_xy.shape)(32, 12, 64, 64, 1)
            #     continue
            
            # # print("count 32 sum_xy.shape=",np.array(sum_xy).shape)
            # # val_ims = sum_xy
            
            # batch_x,batch_y = np.split(sum_xy, 2, axis=1)
            # scaler_path = main_path + 'srandard_scaler/'#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
            # if not os.path.isdir(scaler_path):
            #     os.makedirs(scaler_path)
            # # scaler = load(open('min_max_scaler_4_240210.pkl', 'rb'))       
            # # srandard
            # scaler = load(open(scaler_path+'srandard_scaler_4_240210.pkl', 'rb'))       
            # # scaler = load(open('normalizer_scaler_4_240210.pkl', 'rb'))   
            # batch_x = np.array(batch_x).reshape(-1,64*64)
            # batch_x = scaler.transform(batch_x)
            # '''

            batch_x=np.array(batch_x).reshape(-1,model_parameter['period'],512,512,1)

            
            bothx_y=np.concatenate((batch_x, batch_y), axis=1)
            val_ims = bothx_y
            val_dat = preprocess.reshape_patch(val_ims, args.patch_size)
            print("test_dat  preprocess.reshape_patch=",np.array(val_dat).shape)
            img_gen = model.test(val_dat, real_input_flag)
            print("--預測-")
            print("img_gen model.test=",np.array(img_gen).shape)
            img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
            print("img_gen preprocess.reshape_patch_back=",np.array(img_gen).shape)
            output_length = args.total_length - args.input_length
            # img_gen_length = img_gen.shape[1]
            img_out = img_gen[:, -output_length:]
            test_x_6 = val_ims[:, model_parameter['period']:, :, :, :]
            mse = np.square(test_x_6 - img_out).sum()
            mse_picture_avg = ((mse/args.batch_size)/model_parameter['predict_period'])/(512*512)
            sum_mse_index = sum_mse_index + mse_picture_avg
            # MSE per frame
            seq_p1_cost =0
            sum_mse =0
            avg_seq_p1_cost= 0
            for i in range(output_length):
                x = val_ims[:, i + args.input_length, :, :, :]
                gx = img_out[:, i, :, :, :]
                # gx = np.maximum(gx, 0)
                # gx = np.minimum(gx, 60)
                print("x.shape=",x.shape,"gx=",gx.shape)#x.shape= (4, 64, 64, 1) gx= (4, 64, 64, 1)
                mse = np.square(x - gx).sum()
                img_mse[i] += mse
                avg_mse += mse
                sum_mse +=mse
                account_mse = sum_mse/args.batch_size
                seq_p1_cost = seq_p1_cost + account_mse 
                print("batch_id=",batch_id,"i + args.input_length=",i + args.input_length,"account_mse=",account_mse)
            
                vis_x = x.copy()
                vis_gx = gx.copy()
                vis_x = vis_x/65
                vis_gx = vis_gx/65

                real_frm = np.uint8(vis_x * 255).reshape(512,512)
                pred_frm = np.uint8(vis_gx * 255).reshape(512,512)


                # for b in range(1):
                score, _ = compare_ssim(pred_frm, real_frm, full=True, multichannel=True)
                ssim[i] += score     
                sum_ssim+=score           
            
            avg_seq_p1_cost = ((sum_mse/args.batch_size)/model_parameter['predict_period'])/(512*512)

            batch_cost = batch_cost + avg_seq_p1_cost

        save_path_single_location = save_path+'64x64_val_tw/'
        save_path_index = save_path_single_location + '64x64_PD_index{}/'.format(index)#C:\Users\tilly963\Desktop\predrnn_torch_radar_0915\save_3mitr3_0912\315x315_tes
        avg_ssim = (sum_ssim/model_parameter['predict_period'])/num_of_batch_size
        
        if not os.path.isdir(save_path_index):
            os.makedirs(save_path_index)
        avg_batch_cost =batch_cost/num_of_batch_size
        avg_mse_p1 = sum_mse_index/num_of_batch_size
        print('mse per seq: ' + str(avg_mse))    
        fn = save_path_single_location + 'val_mse_div20_.txt'
        with open(fn,'a') as file_obj:
            file_obj.write("-----val mse itr ="+str(itr)+"----- \n")
            file_obj.write("model_pkl " + model_pkl  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
            file_obj.write("val day len(num_of_batch_size*1) =  " + str(num_of_batch_size*args.batch_size)  + '\n' )
            file_obj.write("place" + str(place)  + '\n' )
            
            # for i in range(args.total_length - args.input_length):
            #     print("sum 64x64 mse seq[",i,"] =",img_mse[i] / (batch_id * args.batch_size))#
            #     print("avg 64x64 mse seq[",i,"] =",((img_mse[i] / (batch_id * args.batch_size))/(64*64))/4)
            #    file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
            file_obj.write("val loss:" + str(avg_batch_cost) + '\n')     
            file_obj.write("val mse:" + str(avg_mse_p1) + '\n')  
            file_obj.write("val avg_ssim:" + str(avg_ssim) + '\n') 
            # file_obj.write("smaple_number:" + str(smaple_number) + '\n')  




        p20_mse = p20_mse+ avg_mse_p1
        p20_cost = p20_cost + avg_batch_cost
    p20_mse = p20_mse/place_len
    avg_p20_cost = p20_cost/place_len
    fn = save_path_single_location + 'val_mse_avg20_.txt'
    with open(fn,'a') as file_obj:
        file_obj.write("-----val mse itr ="+str(itr)+"----- \n")
        file_obj.write("model_pkl " + model_pkl  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
        file_obj.write("val day len(num_of_batch_size*1) =  " + str(num_of_batch_size*args.batch_size)  + '\n' )
        file_obj.write("place 20"   + '\n' )
        
        # for i in range(args.total_length - args.input_length):
        #     print("sum 64x64 mse seq[",i,"] =",img_mse[i] / (batch_id * args.batch_size))#
        #     print("avg 64x64 mse seq[",i,"] =",((img_mse[i] / (batch_id * args.batch_size))/(64*64))/4)
        #   file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
        file_obj.write("val loss:" + str(avg_p20_cost) + '\n') 

        file_obj.write("p20_mse mse:" + str(p20_mse) + '\n')  


    fn = save_path_single_location + 'val_mse_itr.txt'
    with open(fn,'a') as file_obj:
        # file_obj.write("-----test mse itr ="+str(itr)+"----- \n")
        # file_obj.write("model_pkl " + model_name  + '\n args.total_length - args.input_length =' + str(args.total_length - args.input_length)+'\n')
        # file_obj.write("test day len(num_of_batch_size*1) =  " + str(num_of_batch_size*args.batch_size)  + '\n' )
        # file_obj.write("place 20"   + '\n' )
        
        # for i in range(args.total_length - args.input_length):
        #     print("sum 64x64 mse seq[",i,"] =",img_mse[i] / (batch_id * args.batch_size))#
        #     print("avg 64x64 mse seq[",i,"] =",((img_mse[i] / (batch_id * args.batch_size))/(64*64))/4)
        #    file_obj.write('itr:' + str(itr) + ' training loss: ' + str(cost) + '\n')   
        file_obj.write(str(p20_mse) + '\n') 

    fn = save_path_single_location + 'val_ssim_itr.txt'
    with open(fn,'a') as file_obj:
        file_obj.write(str(avg_ssim) + '\n') 
    return p20_mse, avg_ssim
          
if __name__ == "__main__":                
    print('Initializing models')
    
    # from apex import amp
    import torch
    print("================================")
    print(torch.__version__)
    print(torch.version.cuda)
    # print(torch.cuda.amp)
    # print(torch.cuda.amp.autocast)
    model = Model(args)
    gpu_list = np.asarray(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','), dtype=np.int32)
    args.n_gpu = len(gpu_list)
    
    print("args.n_gpu=",args.n_gpu)
    # Env setting
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config) 
    # K.set_session(sess)
    
    # print("cuda = ",config.gpu_options.allow_growth ,"gpu_list=",gpu_list)
    
    
    model_parameter = {"input_shape": [512,512],
                   "output_shape": [512,512],
                   "period": 6,
                   "predict_period": 6}#,
    #               "filter": 36,
    #               "kernel_size": [1, 1]}
     
    # date_date= [['2018-08-23 12:00', '2018-08-23 12:50']]#要大於五筆(5*0.2驗證筆數=1)
    
    ## ['2018-04-23 12:00', '2018-04-23 16:30'],
    # ['2018-06-14 00:00', '2018-06-14 23:50'],
    # date_date= [['2018-06-15 00:00', '2018-06-15 11:00'],
    #             # ['2018-06-15 23:40', '2018-06-15 23:50'],

    #             ['2018-06-16 05:00', '2018-06-16 16:00'],
    #             ['2018-06-16 23:40', '2018-06-16 23:50'],

    #             ['2018-06-17 00:00', '2018-06-17 23:50'],
    #             ['2018-06-18 00:00', '2018-06-18 23:50'],
    #             ['2018-06-19 00:00', '2018-06-19 16:00'],
    #             ['2018-06-19 23:40', '2018-06-19 23:50'],

    #             ['2018-06-20 00:00', '2018-06-20 16:00'],
    #             ['2018-07-01 22:00', '2018-07-01 23:50'],
    #             ['2018-07-02 00:00', '2018-07-02 07:00'],
    #             # ['2018-07-03 22:30', '2018-07-03 23:50'],
    #             # ['2018-07-05 01:00', '2018-07-05 06:00'],
    #             ['2018-07-10 09:30', '2018-07-10 23:50'],
    #             ['2018-07-11 00:00', '2018-07-11 06:00'],
    #             ['2018-08-22 12:00', '2018-08-22 23:50'],
    # date_date= [['2018-06-01 00:00', '2018-08-31 23:59']]
    
    # date_date=[['2017-04-21 10:00', '2017-04-21 23:50']]

    date_date=[['2017-04-21 10:00', '2017-04-21 17:00'],
                ['2017-06-01 14:00', '2017-06-01 23:50'],
                ['2017-06-02 01:00', '2017-06-02 23:00'],
                
                ['2017-06-14 01:00', '2017-06-14 02:00'],
                ['2017-06-14 06:00', '2017-06-14 07:00'],
                
                ['2017-06-16 19:20', '2017-06-16 20:20'],
             
                ['2017-06-18 08:00', '2017-06-18 10:30'],
                
                ['2017-07-29 04:00', '2017-07-29 13:00'],
                ['2017-07-29 23:40', '2017-07-29 23:50'],

                ['2017-07-30 11:00', '2017-07-30 23:50'],
                ['2017-07-31 02:00', '2017-07-31 10:00'],
          
                ['2017-09-29 16:00', '2017-09-29 18:00'],
                ['2017-10-12 22:00', '2017-10-12 23:50'],
                ['2017-10-13 10:00', '2017-10-13 16:00'],
                ['2017-10-14 00:00', '2017-10-14 04:00'],
                ['2017-10-14 20:00', '2017-10-14 23:50'],
              

                ['2018-06-14 12:20', '2018-06-14 12:40'],
                ['2018-06-14 15:00', '2018-06-14 23:50'],
                ['2018-06-15 04:00', '2018-06-15 06:00'],
               
                ['2018-06-17 05:40', '2018-06-17 11:00'],
                ['2018-06-17 22:40', '2018-06-17 22:50'],
                ['2018-06-18 00:50', '2018-06-18 01:50'],#?
                ['2018-06-18 21:00', '2018-06-18 23:50'],#?

                ['2018-06-19 00:00', '2018-06-19 11:00'],
                ['2018-06-20 02:00', '2018-06-20 03:00'],
                ['2018-06-20 13:30', '2018-06-20 14:30'],

                ['2018-07-01 23:00', '2018-07-01 23:40'],
                ['2018-07-03 03:30', '2018-07-02 06:00'],
         
                ['2018-07-10 16:00', '2018-07-10 23:50'],
         
                ['2018-08-22 22:00', '2018-08-22 23:50'],
                ['2018-08-23 00:00', '2018-08-23 23:50'],
                ['2018-08-24 00:00', '2018-08-24 05:00'],

                ['2018-08-24 17:00', '2018-08-24 19:40'],
                ['2018-08-25 16:00', '2018-08-25 16:50'],
                ['2018-08-25 20:40', '2018-08-25 21:40'],

                ['2018-08-26 00:00', '2018-08-26 16:00'],
                ['2018-08-26 18:00', '2018-08-26 21:00'],

                ['2018-08-27 02:00', '2018-08-27 23:50'],
                ['2018-08-28 00:00', '2018-08-28 08:00'],
                ['2018-08-28 15:00', '2018-08-28 23:50'],

                ['2018-08-29 00:00', '2018-08-29 23:50'],
                ['2018-09-08 13:00', '2018-09-08 20:00'],
            
                ['2018-09-15 00:00', '2018-09-15 02:00'],
                ['2018-09-15 10:50', '2018-09-15 12:50'],



                ['2018-09-16 00:00', '2018-09-16 22:00'],#?
                ['2018-10-31 14:00', '2018-10-31 20:00'],
                ['2018-11-02 10:00', '2018-11-02 11:00'],
                
                ['2018-12-23 00:00', '2018-12-23 01:00'],

         
                ['2019-05-17 00:00', '2019-05-17 05:00'],
                ['2019-05-17 09:00', '2019-05-17 10:00'],
                ['2019-05-18 02:30', '2019-05-18 03:30'],
                ['2019-05-18 05:30', '2019-05-18 06:30'],

       
                ['2019-06-10 00:00', '2019-06-10 06:00'],
                ['2019-06-10 16:30', '2019-06-10 23:50'],

                ['2019-06-11 00:00', '2019-06-11 06:00'],
                ['2019-06-11 17:00', '2019-06-11 18:00'],
                ['2019-06-11 20:00', '2019-06-11 21:00'],

                ['2019-06-12 03:30', '2019-06-12 04:00'],
                ['2019-06-12 09:00', '2019-06-12 10:00'],

                ['2019-06-13 09:20', '2019-06-13 09:30'],
                ['2019-06-13 13:30', '2019-06-13 23:00'],


                ['2019-06-14 02:00', '2019-06-14 04:00'],
                ['2019-06-23 00:00', '2019-06-23 10:00'],
                ['2019-06-23 20:00', '2019-06-23 20:10'],
                ['2019-06-23 22:20', '2019-06-23 22:50'],


                ['2019-06-25 03:00', '2019-06-25 05:00'],
                ['2019-06-25 09:00', '2019-06-25 10:00'],

                ['2019-07-18 00:00', '2019-07-18 03:30'],
                ['2019-07-18 12:00', '2019-07-18 13:00'],
                ['2019-07-18 20:00', '2019-07-18 21:00'],


                ['2019-07-19 02:00', '2019-07-19 02:30'],
                ['2019-07-19 12:00', '2019-07-19 12:30'],
                ['2019-07-19 23:40', '2019-07-19 23:50'],

       
                ['2019-07-21 02:00', '2019-07-21 03:30'],
                ['2019-07-21 19:00', '2019-07-21 22:00'],

                ['2019-07-22 00:00', '2019-07-22 02:00'],
             
                ['2019-08-08 04:00', '2019-08-08 06:00'],
                ['2019-08-08 10:00', '2019-08-08 14:00'],
                ['2019-08-08 23:40', '2019-08-08 23:50'],
             
                ['2019-08-09 15:00', '2019-08-09 17:00'],
             
                ['2019-08-13 00:00', '2019-08-13 02:00'],

                ['2019-08-14 03:30', '2019-08-14 03:40'],
                ['2019-08-14 20:00', '2019-08-14 23:00'],
                ['2019-08-14 23:40', '2019-08-14 23:50'],


                ['2019-08-15 04:00', '2019-08-15 04:10'],
                ['2019-08-15 18:00', '2019-08-15 20:00'],
                ['2019-08-15 23:40', '2019-08-15 23:50'],

                ['2019-08-16 04:00', '2019-08-16 05:00'],
                ['2019-08-16 22:00', '2019-08-16 22:30'],

             
        
                ['2019-08-18 04:00', '2019-08-18 15:00'],
                ['2019-08-18 23:00', '2019-08-18 23:50'],

                ['2019-08-19 02:00', '2019-08-19 07:00'],
                ['2019-08-20 08:00', '2019-08-20 08:30'],
                ['2019-08-24 01:00', '2019-08-24 16:00'],
             
                ['2019-09-26 22:00', '2019-09-26 23:50'],
                ['2019-09-27 01:00', '2019-09-27 08:00'],
                ['2019-09-28 06:30', '2019-09-28 07:00'],
               
                ['2019-09-29 18:00', '2019-09-29 23:50'],
                ['2019-09-30 02:00', '2019-09-30 08:00'],
                ['2019-09-30 16:00', '2019-09-30 17:00'],
              
                ['2019-12-29 01:00', '2019-12-29 08:00'],
                ['2019-12-29 15:10', '2019-12-29 15:20'],
                
                ['2019-12-30 04:30', '2019-12-29 06:00']]

                # ['2018-08-24 00:00', '2018-08-24 23:50'],
                # ['2018-08-25 00:00', '2018-08-25 00:20'],
                
                # ['2018-08-25 10:00', '2018-08-25 23:50'],
                # ['2018-08-26 00:00', '2018-08-26 23:50'],
                # ['2018-08-27 00:00', '2018-08-27 23:50'],
                # ['2018-08-28 00:00', '2018-08-28 23:50'],
                # ['2018-08-29 00:00', '2018-08-29 23:50'],
                # ['2018-09-08 00:00', '2018-09-08 01:00'],
                # ['2018-09-08 13:00', '2018-09-08 22:00'],
                # # ['2018-09-09 04:00', '2018-09-09 06:00'],

                # ['2018-09-09 16:00', '2018-09-09 23:00'],
                # ['2018-09-10 05:00', '2018-09-10 14:00'],

                
                # ['2018-09-15 00:00', '2018-09-15 20:00']]
                # ['2018-09-16 00:00', '2018-09-16 22:00'],
                # ['2018-10-31 00:30', '2018-10-31 23:50'],
                # ['2018-11-02 00:00', '2018-11-02 14:00'],
                # ['2018-11-09 22:00', '2018-11-09 23:50'],
                # ['2018-11-10 00:00', '2018-11-10 01:00'],
                # ['2018-11-17 00:00', '2018-11-17 00:20'],
                # ['2018-11-18 15:30', '2018-11-18 16:00'],
                # ['2018-11-25 10:00', '2018-11-25 11:50'],
                # ['2018-11-25 14:40', '2018-11-25 15:00'],

                # ['2018-11-26 02:30', '2018-11-26 04:00'],
                # # ['2018-11-27 16:00', '2018-11-27 23:50'],
                # ['2018-11-28 00:00', '2018-11-28 00:20'],
                # ['2018-12-23 00:00', '2018-12-23 04:30'],
                # ['2018-12-23 21:00', '2018-12-23 21:30']]
    
    # date_date=[['2018-08-23 22:00', '2018-08-23 23:39']]
    #          ['2017-03-10 00:00','2017-03-10 23:59'],
    #          ['2017-04-21 00:00','2017-04-22 23:59'],
    #          ['2017-04-27 00:00','2017-04-27 23:59'],
    #          ['2017-05-16 00:00','2017-05-16 23:59'],
    #          ['2017-05-24 00:00','2017-05-25 23:59'],
    #          ['2017-05-30 00:00','2017-05-30 23:59'],
    #          ['2017-06-02 00:00','2017-06-04 23:59'],
    #          ['2017-06-14 00:00','2017-06-18 23:59'],
    #          ['2017-07-29 00:00','2017-07-31 23:59'],
    #          ['2017-08-22 00:00','2017-08-24 23:59'],
    #         #  ['2017-09-01 00:00','2017-09-02 23:59'],
    #          ['2017-09-29 00:00','2017-09-29 23:59'],
    #          ['2017-10-13 00:00','2017-10-13 23:59'],
    #          ['2017-10-15 00:00','2017-10-15 23:59'],

        
    #          ['2018-01-06 00:00', '2018-01-09 23:59'],
    #          ['2018-05-08 00:00','2018-05-09 23:59'],
    #          ['2018-06-14 00:00','2018-06-20 23:59'],
    #          ['2018-07-02 00:00','2018-07-04 23:59'],
    #          ['2018-07-21 00:00','2018-07-21 23:59'],
    #          ['2018-08-23 00:00','2018-08-30 23:59'],
    #          ['2018-09-08 00:00','2018-09-10 23:59'],
    #          ['2018-09-15 00:00','2018-09-15 23:59'],
    #          ['2018-10-08 00:00','2018-10-10 23:59'],
    #          ['2018-11-02 00:00','2018-11-02 23:59'],
    #          ['2018-12-23 00:00','2018-12-23 23:59'],

    #          ['2019-01-01 00:00', '2019-01-03 23:59'],
    #          ['2019-02-24 00:00','2019-02-24 23:59'],
    #          ['2019-03-08 00:00','2019-03-10 23:59'],
    #          ['2019-04-19 00:00','2019-04-21 23:59'],
    #          ['2019-05-01 00:00','2019-05-01 23:59'],
    #          ['2019-05-15 00:00','2019-05-15 23:59'],
    #          ['2019-05-17 00:00','2019-05-20 23:59'],
    #          ['2019-06-11 00:00','2019-06-14 23:59'],
    #          ['2019-06-23 00:00','2019-06-24 23:59'],
    #          ['2019-07-03 00:00','2019-07-03 23:59'],
    #          ['2019-07-09 00:00','2019-07-10 23:59'],
    #          ['2019-07-19 00:00','2019-07-19 23:59'],
    #          ['2019-08-13 00:00','2019-08-13 23:59'],
    #          ['2019-08-15 00:00','2019-08-20 23:59'],
    #          ['2019-08-24 00:00','2019-08-25 23:59'],
    #          ['2019-09-26 00:00','2019-09-30 23:59'],
    #          ['2019-10-31 00:00','2019-10-31 23:59'],
    #          ['2019-12-05 00:00','2019-12-06 23:59'],
   
    # test_date=[['2017-10-14 00:00', '2017-10-14 00:10'],
    '''
    test_date = [['2018-08-24 00:00','2018-08-24 23:59']]
    '''
    test_date = [['2018-07-24 00:00','2018-07-24 00:19']]

            #  ['2019-05-16 00:00','2019-05-16 23:59'],
            #  ['2019-08-14 00:00','2019-08-14 23:59'],
            #  ['2020-05-27 00:00','2020-05-27 23:59']]
   
    # test_date=[['2018-08-26 00:10', '2018-08-26 00:29']]#]#,#,#]#,
    #          ['2020-05-16 00:00','2020-05-16 23:59'],
    #          ['2020-05-19 00:00','2020-05-19 23:59'],
    #          ['2020-05-21 00:00','2020-05-21 23:59'],
    #          ['2020-05-22 00:00','2020-05-22 23:59'],
    #          ['2020-05-23 00:00','2020-05-23 23:59'],
    #          ['2020-05-26 00:00','2020-05-26 23:59'],
    #          ['2020-05-27 00:00','2020-05-27 23:59'],
    #          ['2020-05-28 00:00','2020-05-28 23:59'],
    #          ['2020-05-29 00:00','2020-05-29 23:59']]
    places=['Sun_Moon_Lake']

    # radar_echo_storage_path= None
    radar_echo_storage_path= 'E:/yu_ting/try/NWP/'#'NWP/'
    load_radar_echo_df_path='InterDST_LSTM_L2_H64_2014to2019_pickday_checkpoint_finetune_modify_loss_d4_v2/2017to2018_pick_day.pkl'

    save_path ='InterDST_LSTM_L2_H64_2014to2019_pickday_checkpoint_finetune_modify_loss_d4_v2/'

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    # radar = load_data(radar_echo_storage_path=radar_echo_storage_path, 
    #                 load_radar_echo_df_path=load_radar_echo_df_path,
    #                 input_shape=model_parameter['input_shape'],
    #                 output_shape=model_parameter['output_shape'],
    #                 period=model_parameter['period'],
    #                 predict_period=model_parameter['predict_period'],
    #                 places=places,
    #                 random=False,
    #                 date_range=date_date,
    #                 test_date=test_date,
    #                 save_np_radar=save_path )
    # if not load_radar_echo_df_path:
        # radar.exportRadarEchoFileList()
        # if not os.path.isdir(save_path):
            # os.makedirs(save_path)
        # radar.saveRadarEchoDataFrame(path=save_path ,load_name_pkl='interdst_over-fitting')   
    
    # sys.exit()
#    save_path = 'save_3m_itr10_0916/'
    # save_path ='save_2y3m_p6_0923_batchsize32_newmodel/'
    
    # save_path ='save_8240110_same_traintest_over_fitting_model_StandardScaler/'

    # sys.exit()
    # save_path ='save_8230110_novaltest_over_fitting_model_itr2000_changing_rate0.0005_model_LayerNormpy_test_1900/'
#    'save_2y3m_0918/'
#    model_name = 'mode_haveLayerNorm_3m_itr3.pkl'
    # model_name = 'mode_haveLayerNorm_2y3m_p6_new_model'
    # model_name = 'model_LayerNormpy_824_Sun_Moon_Lake_model'#'mode_haveLayerNorm_2y3m_p4_new_modelitr7.pkl'

    # model_name = 'model_LayerNormpy_8240110_novaltest_Sun_Moon_Lake_model_itr500.pkl'
    # model_name = 'model_824_Sun_Moon_Lake_itr231.pkl'#'mode_haveLayerNorm_2y3m_p4_new_modelitr7.pkl'
    # model_name = 'model_823_Sun_Moon_Lake_itr1869.pkl'
#    save_path = 'save_2y3m_0918/'
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Normalizer
    # Normalizer().fit(X)
    train_radar_xy_shuffle = None
    val_radar_xy_shuffle = None
    pretrained_model = None#'p1_model_itr26_test_cost6.151229987320058_ssim0.8462856699373386.pkl'#None
    # pretrained_model ='model_itr499_test_cost11.334874471028646_ssim0.641658291434613.pkl' #'p1_model_itr5_test_cost9.519979798220062_ssim0.8119851203114509.pkl'
    
    model_name ='model' #'p1_model_itr5_test_cost9.519979798220062_ssim0.8119851203114509.pkl'

    if args.is_training:
    #    scaler = StandardScaler()
        # muti_sample(save_path,type ='StandardScaler')#!
        # sys.exit()
        # train_radar_xy_shuffle, val_radar_xy_shuffle = load_sample(save_path)#!
        # train_sample_wrapper(model, save_path,pretrained_model , model_name, train_radar_xy_shuffle, val_radar_xy_shuffle)#!
        
        # train_sample_wrapper(model, save_path, model_name, train_radar_xy_shuffle, val_radar_xy_shuffle)#!
        # sys.exit()
        # preprocess_sample(save_path,type ='StandardScaler')
        # preprocess_fun(save_path,type ='StandardScaler')# 'StandardScaler')#!
        # model_name = 'model_8240210_Sun_Moon_Lake_model_itr28914_test_cost0.00023020233493298292.pkl'
        # train_wrapper(model, save_path, model_name )#, train_radar_xy_shuffle, val_radar_xy_shuffle)#!
    #    model_name = 'model_LayerNormpy_8230010_novaltest_Sun_Moon_Lake_model_test_itr1900.pkl'
    #    model_name = 'model_8240110_Sun_Moon_Lake_model_itr15000_test_cost0.560447613398234.pkl'
    #    model_name = 'model_8240110_Sun_Moon_Lake_model_itr10000.pkl'
    #    model_name = 'model_8240110_Sun_Moon_Lake_model_itr15000_test_cost0.026278110841910046.pkl'
    #    model_name = 'model_8240110_Sun_Moon_Lake_model_itr20000_test_cost0.006796003629763921.pkl'
    #    model_name ='model_8240110_Sun_Moon_Lake_model_itr30000_test_cost0.0007234304212033749.pkl'
        # model_name ='model_8240210_0310_Sun_Moon_Lake_model_itr34597_test_cost0.00037630179819340504.pkl'
        # model_name = 'model_8240210_0310_Sun_Moon_Lake_model_itr185_test_cost3.1776345775001946.pkl'
        model_name = 'model_loss_itr8_test_cost27.89335529124679.pkl'
        # model_name = '.9234249002449655_ssim0.8543377402331324.pkl'
        # test_wrapper(model, save_path, model_name,itr=26,load_model=True)
      
        test_show(model, save_path, model_name,itr=8)
        # for i in range(1,21):
        #     model_name = 'p1_model_itr{}.pkl'.format(i)
        #     test_wrapper(model, save_path, model_name,itr=i,load_model=True)
    
    # else:
    #   val(model, save_path,model_pkl = model_name, itr=10)
        # tw(model, save_path, model_name, itr=1)


        # for period in range(model_parameter['predict_period']):
        #     #    print('pred_y[:, period] = ', pred_y[:, period])
        #     #    print('test_y[:, period] = ', test_y[:, period])
        #     csi_eva = Verification(pred=pred_y[:, period].reshape(-1, 1), target=test_y[:, period].reshape(-1, 1), threshold=60, datetime='')
        #     csi.append(np.nanmean(csi_eva.csi, axis=1))
        
        # csi = np.array(csi)
        # np.savetxt(save_path+'T202005270000csi_reshape1.csv', csi, delimiter = ',')
        # np.savetxt(save_path+'T202005270000csi.csv', csi.reshape(6,60), delimiter = ' ')

        # ## Draw thesholds CSI
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        # ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        # plt.xlim(0, 60)
        # plt.ylim(-0.05, 1.0)
        # plt.xlabel('Threshold')
        # plt.ylabel('CSI')
        # plt.title('20200527day\nThresholds CSI')
        # plt.grid(True)

        # all_csi = []
        # for period in range(model_parameter['predict_period']):
        # plt.plot(np.arange(csi.shape[1]), [np.nan] + csi[period, 1:].tolist(), 'o--', label='{} min'.format((period+1)*10))

        # plt.legend(loc='upper right')

        # fig.savefig(fname=save_path+'Thresholds_CSI.png', format='png')
        # plt.clf()


        # ## Draw thesholds AVG CSI
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        # ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        # plt.xlim(0, 60)
        # plt.ylim(-0.05, 1.0)
        # plt.xlabel('Threshold')
        # plt.ylabel('CSI')
        # plt.title('20200527\nThresholds CSI')
        # plt.grid(True)

        # all_csi = []
        # plt.plot(np.arange(csi.shape[1]), [np.nan] + np.mean(csi[:, 1:], 0).tolist(), 'o--', label='AVG CSI')
        
        # plt.legend(loc='upper right')

        # fig.savefig(fname=save_path+'Thresholds_AVG_CSI.png', format='png')
        # plt.clf()

        # #csie = time.clock()
        # #
        # #alle = time.clock()
        # #
        # #print("load NWP time = ", loadNe - loadNs)
        # #print("load CREF time = ", loadCe - loadCs)
        # #print("All time = ", alle - alls)
        # ## Draw peiod ALL CSI 
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        # ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
        # plt.xlim(0, model_parameter['predict_period']+1)
        # plt.ylim(-0.05, 1.0)
        # plt.xlabel('Time/10min')
        # plt.ylabel('CSI')
        # my_x_ticks = np.arange(0, model_parameter['predict_period']+1, 1)
        # plt.xticks(my_x_ticks)
        # plt.title('Threshold 5-55 dBZ')
        # plt.grid(True)
        # i = 0
        # for threshold in range(5, 56, 5):
        # plt.plot(np.arange(len(csi)+1), [np.nan] + list(csi[:, threshold-1]), 'o--', label='{} dBZ'.format(threshold), color=Color[i])
        # i = i + 1
        # #plt.legend(loc='lower right')

        # plt.clf()

        # fig.savefig(fname=save_path+'Period_CSI_ALL2.png', format='png')


        # rmse=np.sqrt(((pred_list - test_y) ** 2).mean())
        # fn = save_path + 'p20_20200527_rmse.txt'
        # with open(fn,'a') as file_obj:
        #     file_obj.write('rmse=' + str(rmse)+'\n')

        #     # file_obj.write('mse=' + str(mse)+'\n')