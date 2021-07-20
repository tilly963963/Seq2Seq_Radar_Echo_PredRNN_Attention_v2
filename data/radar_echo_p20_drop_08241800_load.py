## Other libs
import os, sys
import tarfile, gzip, struct, shutil
import pandas as pd
import numpy as np
import pickle as pkl
from time import time 
from datetime import datetime, timedelta
from enum import Enum
import pickle as pkl

## ML libs
import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

## Custom libs
from area_20 import area_20#dic name
from CustomUtils import ShowProcess

"""
    Classes:
     
        fileformat   輸入資料格式列舉
        load_data    載入雷達回波資料，並處理成序列並存於DataFrame中
        generator    資料生成器，用於將資料載入模型
    
    load_data Parameter:

        radar_echo_storage_path    雷達回波原始資料儲存路徑
        input_shape                輸入矩陣大小
        output_shape               輸出矩陣大小
        period                     輸入時間序列長度
        predict_period             輸出時間序列長度
        place                      訓練資料地點
        date_range                 訓練資料時間範圍  ex.[['2017-01-01 00:00', '2017-11-30 23:59'],
                                                      ['2018-05-01 00:00', '2018-11-30 23:59']]
        test_date                  測試日期 ex. [['2018-08-23 00:00', '2018-08-30 23:59']]
        val_split                  驗證集比例
        random                     資料順序是否打亂
        random_seed                隨機種子
        radar_echo_name_format     雷達回波原始資料檔案名稱格式
        radar_echo_file_format     雷達回波原始資料檔案格式
        load_radar_echo_df_path    雷達回波處理後DataFrame儲存路徑
        
"""

class fileformat(Enum):
    GZ = '.gz'
    NPY = '.npy'
    NONE = ''
'''
        'Banqiao':area_20_466880,
        'Keelung':area_20_466940,
        'Taipei':area_20_466920,
        'New_House':area_20_467050,
        'Chiayi':area_20_467480,
        'Dawu':area_20_467540,
        'Hengchun':area_20_467590,
        'Success':area_20_467610,
        'Sun_Moon_Lake':area_20_467650,
        'Taitung':area_20_467660,
        'Yuxi':area_20_467770,
        'Hualien':area_20_466990,
        'Beidou':area_20_C0G840,
        'Bao_Zhong':area_20_C0K430,
        'Chaozhou':area_20_C0R220,
        'News':area_20_C0R550,
        'Member_Hill':area_20_C0U990,
        'Yuli':area_20_C0Z061,
        'Snow_Ridge':area_20_C1F941,
        'Shangdewen':area_20_C1R120,
'''

class load_data(object):
    def __init__(self, radar_echo_storage_path,
                       input_shape=[21, 21],
                       output_shape=[1, 1],
                       period=6, 
                       predict_period=18,
                       places=['Banqiao','Keelung','Taipei','New_House','Chiayi',
                               'Dawu','Hengchun','Success','Sun_Moon_Lake','Taitung',
                               'Yuxi','Hualien','Beidou','Bao_Zhong','Chaozhou',
                               'News','Member_Hill','Yuli','Snow_Ridge','Shangdewen'],

#                       date_range=[['2017-01-01 00:00', '2017-11-30 23:59'],
#                                   ['2018-05-01 00:00', '2018-11-30 23:59']],
                       date_range="",
                       test_date="",
                       val_split=0.2,
                       random=True,
                       random_seed=45,
                       radar_echo_name_format='COMPREF.%Y%m%d.%H%M',
                       radar_echo_file_format=fileformat.GZ,
                       load_radar_echo_df_path=None):
        self._places = places
        self._places_dict = {}
        self._radar_echo_storage_path = radar_echo_storage_path
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._period = period
        self._predict_period = predict_period
        
        self._date_range = date_range
        self._test_date = test_date
        self._val_split = val_split
        self._random = random
        self._random_seed = random_seed
        self._radar_echo_name_format = radar_echo_name_format
        self._radar_echo_file_format = radar_echo_file_format
        self._load_radar_echo_df_path = load_radar_echo_df_path
        self._file_list = None
        self._date_ranged = self._buildSourceDateRange()
        self._initPlaceLatLontoXY(places)#,place3,place4,place5,place6,place7)
        self._createRadarEchoDict()
        self._createDataSetSequence()
        self._testValDatetimeExtract()


    def _initPlaceLatLontoXY(self,places):#,place3,place4,place5,place6,place7):
        """Transfor place (longitude, latitude) to matrix (X, Y)"""
        
        for i in range(len(places)):
            #print("place is ",places[i])
            self._place = places[i]
            lat = area_20[self._place].lat
            lon = area_20[self._place].lon
            self._y = int(881 - np.ceil((29.0125 - lat)/0.0125))
            self._x = int(np.ceil((lon - 115.0)/0.0125))
            self._places_dict[places[i]]={'y':self._y,'x':self._x}
        # print(self._places_dict)
        for  place,xy in self._places_dict.items():
            print("place in",place,"x is",xy['x'],'y is',xy['y'])
        return self._places_dict


    def _buildDateRange(self, date_range):
        """Turn string format datetime to datetime format list"""
        date_range_temp = []
        for date in date_range:
            date_range_temp += pd.date_range(date[0], date[1], freq='10T').tolist()
        
        return date_range_temp

    def _buildSourceDateRange(self):
        """Turn string format datetime to datetime format list for source data"""
        date_range = np.vstack((self._date_range, self._test_date))
        date_range_temp = []
        
        for date in date_range:
            date_start = datetime.strptime(date[0], "%Y-%m-%d %H:%M") - timedelta(minutes=10*self._period)
            date_end = datetime.strptime(date[-1], "%Y-%m-%d %H:%M") + timedelta(minutes=10*(self._predict_period-1))
            date_range_temp += pd.date_range(date_start, date_end, freq='10T').tolist()
        date_range_temp = list(dict.fromkeys(date_range_temp))
        return date_range_temp

    def _buildRadarEchoFileList(self):
        """Build Radar Echo file list"""
        file_list_temp = []
        print("Building Radar Echo file list...")
        if os.path.isdir(self._radar_echo_storage_path):
            for root, dirs, files in os.walk(self._radar_echo_storage_path):
                for f in files:
                    fullpath = os.path.join(root, f)
                    file_list_temp.append(fullpath.replace("\\", "/"))
            
            file_list = []
            sp = ShowProcess(len(file_list_temp))
            for filepath in file_list_temp:
                sp.show_process()
                try:
                    date = datetime.strptime(filepath.split('/')[-1], self._radar_echo_name_format+self._radar_echo_file_format.value)
                    if date in self._date_ranged:
                        file_list.append([date.strftime("%Y%m%d.%H%M"), filepath])
                except:
                    continue
        else:
            print("Radar Echo Storage Path Error!")
            exit()

        self._file_list = file_list
        print("Build Radar Echo file list finished!")
    
    def _createRadarEchoDict(self):
        """Create Radar Echo file list dict index by date"""
        if self._load_radar_echo_df_path:
            self._radar_echo_df = pd.read_pickle(self._load_radar_echo_df_path)
            return 0

        self._buildRadarEchoFileList()
        
        dict_date = []
        dict_path = []
        print("Creating Radar Echo dataframe...")

        sp = ShowProcess(len(self._file_list))
        for val in self._file_list:
            sp.show_process()
            dict_date.append(val[0])
            dict_path.append(val[1])

        self._radar_echo_df = pd.DataFrame({"datetime": pd.to_datetime(dict_date, format='%Y%m%d.%H%M'),
                                            "path": dict_path}).drop_duplicates(subset='datetime', keep='first').sort_values('datetime').set_index('datetime')
        

        print("Radar Echo loading...")
        sp = ShowProcess(len(self._radar_echo_df['path']))
#        
#                for  place,xy in places_dict.items():
#            print("place in"+place+"x is"+xy['x']+'y is'+xy['y'])
        i=1    
        for  place,xy in self._places_dict.items():
            radar_echo = []
            destory_datetime = []            
            print("place in",place,"x is",xy['x'],'y is',xy['y'])
            for idx, path in zip(self._radar_echo_df.index, self._radar_echo_df['path']):
            #            sp.show_process()
                try:
                    radar_echo.append(self._radarEchoUnpack(path,xy['x'],xy['y']))
                except:
                    radar_echo.append([])   
            self._radar_echo_df.insert(i, place, radar_echo)
            self._radar_echo_df.drop([val[0] for val in destory_datetime])                
            i+=1
            print("self._radar_echo_df=")
            print(self._radar_echo_df)           
        print("self._radar_echo_df1.shape=",self._radar_echo_df.shape)        
        print("self._radar_echo_df1.info()=",self._radar_echo_df.info())    
        print("##########")

        print("destory_datetime=")
        print(destory_datetime)
        for val in destory_datetime:
            with open(self._radar_echo_storage_path + "destory_file.txt", 'w') as destory:
                destory.write(val[0].strftime("%Y-%d-%m %H:%M") + ', ' + val[1] + '\n')
        
    def _createDataSetSequence(self):
        """Processing data sequence for each datetime"""
        radar_date = []
        print("Data Sequence Checking...")
        sp = ShowProcess(len(self._date_ranged))
        for dt in self._date_ranged:
#            sp.show_process()
            checked, idx_list = self._datetimeSequenceCheck(dt)
            if checked:
                radar_date.append([dt, idx_list])
        print("Checking finished!")
        self._radar_echo_sequence_df = pd.DataFrame({'datetime': [val[0] for val in radar_date],
                                                     'RadarEchoSequence': [val[1] for val in radar_date]}).set_index('datetime').sort_index()
        # date_drop = pd.date_range(['2018-08-24 18:00:00'],end=['2018-08-24 18:00:00'], freq='0T')
        # datetime.strptime(['2018-08-24 18:00:00'], "%Y-%m-%d %H:%M")
        # self._radar_echo_sequence_df.drop(datetime.strptime('2018-08-24 18:00', "%Y-%m-%d %H:%M"))
        # self._radar_echo_sequence_df.to_excel("train_805.xlsx")
        # # sys.exit()
        # self._radar_echo_sequence_df=self._radar_echo_sequence_df[self._radar_echo_sequence_df['datetime']!='2018-08-24 18:00:00']
        self._radar_echo_sequence_df.to_excel("train_drop1800.xlsx")

    def _datetimeSequenceCheck(self, dt):
        """Check data sequnce completeness"""
        print("dt=",dt)
        # print("type(dt)=",type(dt))
        date_start = dt - timedelta(minutes=10*self._period)
        date_end = dt + timedelta(minutes=10*(self._predict_period-1))
        daterange = pd.date_range(date_start, end=date_end, freq='10T')
        if dt == datetime.strptime('2018-08-24 18:00', "%Y-%m-%d %H:%M"):
            print("@@@@@@@@@@@@@")
            print("dt drop",dt)
            return False, []
        for date in daterange:
            if not date in self._radar_echo_df.index:
                return False, []
        return True, daterange

    def _radarEchoUnpack(self, file_path, x, y):
        """Load Radar Echo from file"""
 #       print("!!!!!")
#        print("x=",x,"y=",y)   
        radar=[]
        if self._radar_echo_file_format == fileformat.GZ:
            data = gzip.open(file_path).read()
            radar = struct.unpack(881*921*'h', data[-881*921*2:])

        elif self._radar_echo_file_format == fileformat.NONE:
            with open(file_path, 'rb') as d:
                data = d.read()
                radar = struct.unpack(881*921*'h', data[-881*921*2:])
        radar_data=[]
        radar = np.array(radar).reshape(881, 921).astype(np.float32)/10
        # print("1_floa64 radar.dtype=",radar.dtype)  
        radar = radar.astype(np.float32)
        # print("2_float32 radar.dtype=",radar.dtype)  
        radar_data = radar[y-int(self._input_shape[1]/2):y+int(self._input_shape[1]/2), x-int(self._input_shape[0]/2):x+int(self._input_shape[0]/2)]
#        radar_data.append(radar[self._y2-int(self._input_shape[1]/2):self._y2+int(self._input_shape[1]/2)+1, self._x2-int(self._input_shape[0]/2):self._x2+int(self._input_shape[0]/2)+1])
#        radar_data.append(radar[self._y3-int(self._input_shape[1]/2):self._y3+int(self._input_shape[1]/2)+1, self._x3-int(self._input_shape[0]/2):self._x3+int(self._input_shape[0]/2)+1])
#        radar_data.append(radar[self._y4-int(self._input_shape[1]/2):self._y4+int(self._input_shape[1]/2)+1, self._x4-int(self._input_shape[0]/2):self._x4+int(self._input_shape[0]/2)+1])
#        radar_data.append(radar[self._y5-int(self._input_shape[1]/2):self._y5+int(self._input_shape[1]/2)+1, self._x5-int(self._input_shape[0]/2):self._x5+int(self._input_shape[0]/2)+1])
#        radar_data.append(radar[self._y6-int(self._input_shape[1]/2):self._y6+int(self._input_shape[1]/2)+1, self._x6-int(self._input_shape[0]/2):self._x6+int(self._input_shape[0]/2)+1])
#        radar_data.append(radar[self._y7-int(self._input_shape[1]/2):self._y7+int(self._input_shape[1]/2)+1, self._x7-int(self._input_shape[0]/2):self._x7+int(self._input_shape[0]/2)+1])
#                    print("float64 batch_x.dtype=",batch_x.dtype)
        # print("3_floa64 radar_data.dtype=",radar_data.dtype)  
        radar_data = radar_data.astype(np.float32)
        # print("4_float32 radar_data.dtype=",radar_data.dtype)  
        radar_data = radar_data.flatten()
#        print("################################")
 #       print("radar_data.shape=",radar_data.shape)
        radar_data = [val if val > 0.0 else 0.0 for val in radar_data]
        radar_data = np.array(radar_data).reshape(self._input_shape)
#        print("radar_data.shape=",radar_data.shape)
#        data.close()
#        if data.readline():
#            print("close")
#            data.close()
#        if file_path == 'NWP/compref_mosaic/COMPREF.20180827.2350.gz':
#            data.close()
        return radar_data

    def _testValDatetimeExtract(self):
        "Extract test datetime and val datetime from Radar Echo Sequence dataframe"
        test_date_range = self._buildDateRange(self._test_date)
        
        test_date_range_temp_seq = []
        test_date_range_temp = []
        for date in test_date_range:
            if date in self._radar_echo_sequence_df.index:
                test_date_range_temp.append(date)
                test_date_range_temp_seq.append(date)

        self._test_radar_echo_seq_df = pd.DataFrame({'datetime': test_date_range_temp,
                                                     'RadarEchoSequence': self._radar_echo_sequence_df['RadarEchoSequence'][test_date_range_temp_seq]})
        print("self._test_radar_echo_seq_df.info()=")
        print(self._test_radar_echo_seq_df.info())
        radar_echo_seq_df = self._radar_echo_sequence_df.drop(test_date_range_temp).reset_index()

        if self._random:
            radar_echo_seq_df = shuffle(radar_echo_seq_df)

        self._train_radar_echo_seq_df = radar_echo_seq_df.iloc[:int(-(self._val_split)*len(radar_echo_seq_df))].set_index('datetime')
        self._val_radar_echo_seq_df = radar_echo_seq_df.iloc[int(-(self._val_split)*len(radar_echo_seq_df)):].set_index('datetime')
        print("self._train_radar_echo_seq_df=")
        print(self._train_radar_echo_seq_df)
        print("self._val_radar_echo_seq_df=")
        print(self._val_radar_echo_seq_df) 
        print("self._test_radar_echo_seq_df=")

        print(self._test_radar_echo_seq_df)
    def saveConfig(self, save_path='data/'):
        """Save Config"""
        config_dict = {'radar_echo_storage_path': self._radar_echo_storage_path,
                       'input_shape': self._input_shape,
                       'output_shape': self._output_shape,
                       'period': self._period,
                       'predict_period': self._predict_period,
                       'place1': self._place1,
                       'place2': self._place2,
                       'place3': self._place3,
                       'place4': self._place4,
                       'place5': self._place5,
                       'place6': self._place6,
                       'place7': self._place7,
                       'date_range': self._date_range,
                       'test_date': self._test_date,
                       'test_split': self._val_split,
                       'random': self._random,
                       'random_seed': self._random_seed,
                       'load_radar_echo_df_path': self._load_radar_echo_df_path,
                       'radar_echo_name_format': self._radar_echo_name_format,
                       'radar_echo_file_format': self._radar_echo_file_format}

        with open(save_path + 'config.pkl', 'wb') as cfg:
            pkl.dump(config_dict, cfg)

    def exportRadarEchoFileList(self, save_path=None):
        """Save Radar Echo file list"""
        if not save_path:
            save_path = self._radar_echo_storage_path
        if not self._file_list:
            self._buildRadarEchoFileList()

        print("Saving...")
        sp = ShowProcess(len(self._file_list))
        with open(save_path + 'RadarEchoFileList.txt', 'w') as filelist:
            filelist.write("datetime, radarfilepath\n")
            for path in self._file_list:
                sp.show_process()
                filelist.write(path[0] + ', ' + path[1] + '\n')
        print("Radar Echo File List Saved in \'" + self._radar_echo_storage_path + "\'")

    def saveRadarEchoDataFrame(self, path='data/',load_name_pkl='default'):
        """Save Radar Echo dataframe"""
        self._radar_echo_df_path = path + '{}.pkl'.format(load_name_pkl)
#        self._radar_echo_df_path = path + '{}_RadarEcho_{}x{}.pkl'.format(self._place1, self._place2, self._place3, self._place4, self._place5, self._place6, self._place7, self._input_shape[0], self._input_shape[1])
        self._radar_echo_df.to_pickle(self._radar_echo_df_path)

    # @property
    # def getRadarEchoDataFramePath(self):
    #     """Return radar echo pickle file storage path"""
    #     return self._radar_echo_df_path
    
    def getRadarEchoDataFrame(self):
        """Return Radar Echo dataframe"""
        return self._radar_echo_df

    def getRadarEchoSequenceDataFrame(self):
        """Return Radar Echo sequence dataframe"""
        return self._radar_echo_sequence_df

    def generator(self, type='train', batch_size=32):
        """Create generator"""
        if type == 'train':
            return generator(radar_echo_dataframe=self._radar_echo_df,
                             radar_echo_seq_dataframe=self._train_radar_echo_seq_df,
                             input_shape=self._input_shape,
                             output_shape=self._output_shape,
                             period=self._period,
                             predict_period=self._predict_period,
                             batch_size=batch_size,
                             random=self._random,
                             places=self._places,
                             places_dict=self._places_dict)

        if type == 'val':
            return generator(radar_echo_dataframe=self._radar_echo_df,
                             radar_echo_seq_dataframe=self._val_radar_echo_seq_df,
                             input_shape=self._input_shape,
                             output_shape=self._output_shape,
                             period=self._period,
                             predict_period=self._predict_period,
                             batch_size=batch_size,
                             random=self._random,
                             places=self._places,
                             places_dict=self._places_dict)

        if type == 'test':
            return generator(radar_echo_dataframe=self._radar_echo_df,
                             radar_echo_seq_dataframe=self._test_radar_echo_seq_df,
                             input_shape=self._input_shape,
                             output_shape=self._output_shape,
                             period=self._period,
                             predict_period=self._predict_period,
                             batch_size=batch_size,
                             random=False,#!
                            #  random=self._random,
                             places=self._places,
                             places_dict=self._places_dict)

        return None

    def getRadarEchobyDatetime(self, start, end):
        return 
    
    def getDataSetbyDatetime(self, datetimes):
        return 


class generator(keras.utils.Sequence):
    def __init__(self, radar_echo_dataframe,
                       radar_echo_seq_dataframe,
                       input_shape=[105, 105],
                       output_shape=[1, 1],
                       period=6,
                       predict_period=6,
                       batch_size=32,
                       random=True,
                       places=None,
                       places_dict=None):
        self._places = places
        self._places_dict = places_dict
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._period = period
        self._predict_period = predict_period
        self._radar_echo_dataframe = radar_echo_dataframe
        self._radar_echo_seq_dataframe = radar_echo_seq_dataframe
        self._data_shape = np.array(radar_echo_dataframe[places[0]].values[0]).shape
        print("self._data_shape=",self._data_shape)
        self._batch_size = batch_size
        self._step_per_epoch = self.__len__()
        self._steps = np.arange(self._step_per_epoch)
        self._random = random
#        if self._random:
#            self._steps = shuffle(self._steps)

    def generator_getClassifiedItems_3(self, index, place):
        """Return batch X, y sequence data"""
        print("***int radar echo generator_getClassifiedItems***")
        idx = self._steps[index]        
        batch_X = []
        batch_y = []
#        print("index=",index,"idx=",idx)
        print("range(idx*self._batch_size, (idx+1)*self._batch_size)",range(idx*self._batch_size, (idx+1)*self._batch_size))
        # for place in range(len(self._places)):
        for batch in range(idx*self._batch_size, (idx+1)*self._batch_size):
            try:
                radar_echo = np.array(self._radar_echo_dataframe[place][self._radar_echo_seq_dataframe['RadarEchoSequence'][batch]].tolist())
                batch_X.append(self._getCentral(radar_echo[:self._period], self._input_shape).reshape([self._period]+self._input_shape+[1]))
#                print("np.array(radar_echo[-1]).shape=",np.array(radar_echo[-1]).shape)
                # batch_y.append(self._getCentral(radar_echo[-self._predict_period:], self._output_shape).reshape([1*self._output_shape[0]*self._output_shape[1]]))
                batch_y.append(self._getCentral(radar_echo[-self._predict_period:], self._output_shape).reshape([self._predict_period*self._output_shape[0]*self._output_shape[1]]))
                
                print("np.array(batch_X), np.array(batch_y).shape",np.array(batch_X).shape, np.array(batch_y).shape)

            except:
                return np.array(batch_X), np.array(batch_y)  
            if len(batch_X)==1:
                print("np.array(batch_X), np.array(batch_y).shape",np.array(batch_X).shape, np.array(batch_y).shape)
        return np.array(batch_X), np.array(batch_y)
    
    def __len__(self):
        """Return batch num"""
        return int(np.ceil(len(self._radar_echo_seq_dataframe)/self._batch_size))
    
    @property
    def step_per_epoch(self):
        """Retuen step per epoch"""
        return self._step_per_epoch

    def on_epoch_end(self):
        """If random index at epoch end"""
        if self._random:
            self._steps = shuffle(self._steps)

    def _getCentral(self, data, shape):
        """get the data Central"""
        # print("self._data_shape[0]",self._data_shape[0],"shape[0]=",shape[0])
        if self._data_shape[0] < shape[0] or self._data_shape[1] < shape[1]:
            print("Data shape is not big enough. \nPlease reprocess radar echo datafram using load_data class.")
            exit()
        
        if self._data_shape[0] == shape[0] and self._data_shape[0] == shape[0]:
            return data
            
        return data[:, self._data_shape[0]//2-shape[0]//2:self._data_shape[0]//2+shape[0]//2+1, self._data_shape[1]//2-shape[1]//2:self._data_shape[1]//2+shape[1]//2+1]


def loadConfig(path):
    """Load config from path"""
    with open(path, 'rb') as cfg:
        config_dict =pkl.load(cfg)
    return config_dict

if __name__ == "__main__":
    
    from time import time
    start = time()
    data = load_data(radar_echo_storage_path='NWP/', 
                     load_radar_echo_df_path='data/Pingtung_RadarEcho_21x21.pkl' ,
                     input_shape=[21, 21],
                     output_shape=[1, 1],
                     period=6,
                     predict_period=18,
                     place1='Pingtung',
                     place2='Taipei',
#                     place3='Taoyuan',
#                     place4='Taichung',
#                     place5='Kaohsiung',
#                     place6='Miaoli',
#                     place7='BaoZhong',
                     date_range=[['2018-08-26 00:00', '2018-08-26 23:59']],
                     test_date=[['2018-08-27 00:00', '2018-08-27 23:59']])

    end = time()
    data.exportRadarEchoFileList()
    data.saveRadarEchoDataFrame()
    

    df = data.getRadarEchoDataFrame()
    print(df.head(5))

    radarseq = data.getRadarEchoSequenceDataFrame()
    print(radarseq.head(5))

    train_generator = data.generator('train')
    print(train_generator.step_per_epoch)

    val_generator = data.generator('val')
    print(val_generator.step_per_epoch)

    test_generator = data.generator('test')
    print(test_generator.step_per_epoch)

    print("Cost %.2f" % (end-start))

    for idx in range(train_generator.step_per_epoch):
        batch_X, batch_y = train_generator.__getitem__(idx)
        print(np.array(batch_X).shape)
        print(np.array(batch_y).shape)
