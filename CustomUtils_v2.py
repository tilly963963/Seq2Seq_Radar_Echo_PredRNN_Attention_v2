import os, sys, time
import numpy as np
from area import area 
import numpy as np
from data.radar_echo_k3_p20 import load_data, generator
'''
def generator_getClassifiedItems(clf, generator, classified,places):
    temp_X = []
    temp_y = []
    print("range(generator.step_per_epoch)=",range(generator.step_per_epoch))
    for place in range(len(places)):
        print("########",places[place],"########")
        for idx in range(generator.step_per_epoch):
            print("idx=",idx)
            batch_x, batch_y = generator.__getitem__(idx, places[place])
            print("batch_x=",batch_x.shape,"batch_y=",batch_y.shape)
            #batch_x= (4, 6, 55, 55, 1) batch_y= (4, 6)
            for i, val in enumerate(batch_x):#32
                # print("val=",np.array(val).shape)#val= (6, 55, 55, 1)
                c = clf.predict([val.flatten()])[0]
                if classified == c:
                    temp_X.append(batch_x[i])
                    temp_y.append(batch_y[i])
    print("np.array(temp_X).shape=",np.array(temp_X).shape)# (1995, 6, 55, 55, 1)
    print("np.array(temp_y).shape=",np.array(temp_y).shape)
    return np.array(temp_X), np.array(temp_y)
 '''   

def generator_getClassifiedItems(generator, places):
    temp_X = []
    temp_y = []
    print("====generator_getClassifiedItems===")
    print("places=",places)
    print("range(generator.step_per_epoch)=",range(generator.step_per_epoch))
    for place in range(len(places)):
        print("########",places[place],"########")
        for index in range(generator.step_per_epoch):
            # print("places[place]=",places[place])
            
            batch_x, batch_y = generator.generator_getClassifiedItems_3(index, places[place])
            print("batch_x=",batch_x.shape,"batch_y=",batch_y.shape)
            # print("float64 batch_x.dtype=",batch_x.dtype)
            batch_x = batch_x.astype(np.float16)
            # print("float32 batch_x.dtype=",batch_x.dtype)
            
            batch_y = batch_y.astype(np.float16)
            print("batch_x=",batch_x.shape,"batch_y=",batch_y.shape)

            # temp_X.append()
            #batch_x= (4, 6, 55, 55, 1) batch_y= (4, 6)
            for i, val in enumerate(batch_x):#32
                # print("val=",np.array(val).shape)#val= (6, 55, 55, 1)
                # c = clf.predict([val.flatten()])[0]
                # if classified == c:
                temp_X.append(batch_x[i])
                temp_y.append(batch_y[i])
    print("np.array(temp_X).shape=",np.array(temp_X).shape)# (1995, 6, 55, 55, 1)
    print("np.array(temp_y).shape=",np.array(temp_y).shape)
    return np.array(temp_X), np.array(temp_y)
    
def PathCheck(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    
def SaveSummary(model, path):
    with open(path + '/report.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
# 
def get_xy(place=None):
    
    lat = area[place].lat
    lon = area[place].lon
    x = int(np.ceil((lon - 115.0)/0.0125))
    y = int(881 - np.ceil((29.0125 - lat)/0.0125))

    return x, y


class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0 # 当前的处理进度
    max_steps = 0 # 总共需要处理的次数
    max_arrow = 50 #进度条的长度
    infoDone = 'Done'

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps, infoDone=None):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        if self.infoDone:
            print(self.infoDone)
        self.i = 0

