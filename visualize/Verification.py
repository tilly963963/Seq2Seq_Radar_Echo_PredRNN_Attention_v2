import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors


"""
    Class:
        Verification

    Verification Parameter:

        pred        模型雷達回波預測結果
        target      實際雷達回波值
        datetime    資料時間範圍
        title       圖表顯示標題    
        threshold   雷達回波門檻值範圍
 
"""

class Verification(object):
    def __init__(self, pred, target, datetime=[], title='', threshold=60):
        self.pred = pred#(筆數,預測的010~060)(144,1)
        self.target = target
        self.threshold = threshold
        self.datetime = datetime
        self.title = title
        # self.csi, self.far, self.pod, self.acc = self.confusion_matrix()
        self.csi = self.confusion_matrix()

    def __CSI__(self, a, b, c):
        # print("a=",a,"b=",b,"c=",c)
        if a+b+c == 0:
            # print("return None ")
            
            return None
        # print("return = ",str(np.round(a/(a+b+c), 3)))
        # np.float32(1e35)
            # batch_y = batch_y.astype(np.float16)
        ans = np.round(a/(a+b+c), 3)
        # print("1 floa32 ans.dtype=",ans.dtype)  

        ans = ans.astype(np.float16)
        # print("2 floa16 ans.dtype=",ans.dtype)  
        return ans

        # return np.round(a/(a+b+c), 3)
    def __FAR__(self, a, b):
        if a+b == 0:
            return None
        return np.round(b/(a+b), 3)

    def __POD__(self, a, c):
        if a+c == 0:
            return None
        return np.round(a/(a+c), 3)

    def __ACC__(self, a, b, c, d):
        if a+b+c+d == 0:
            return None
        return np.round((a+d)/(a+b+c+d), 3)

    def _visualize(self, kinds, title):
        n = len(kinds)
        start = self.datetime[0].strftime("%Y%m%d-%H%M")
        end = self.datetime[-1].strftime("%Y%m%d-%H%M")
        date_ticks = [self.datetime[i].strftime(
            "%H:%M") for i in range(0, len(self.datetime), 6)]
        x_range = [i for i in range(0, len(self.datetime), 6)]
        y_range = [i for i in range(1, self.threshold, 10)]

        if n == 1:
            fig, ax = plt.subplots(n, 1, figsize=(10, 5*n))
            for idx in range(n):
                bounds = np.round(np.linspace(0.0, 1.0, 21), 2)
                norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                im = ax.pcolormesh(kinds[idx], norm=norm, cmap='jet')
                ax.set_facecolor((0.8, 0.8, 0.8))
                # ax.title.set_text("{}\n{} to {}".format(
                #     title[idx], str(start), str(end)))
                ax.title.set_text(self.title)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2.5%", pad=0.05)
                plt.colorbar(im, ax=ax, cax=cax, ticks=bounds)
                ax.invert_yaxis()
                plt.sca(ax)
                plt.subplots_adjust(bottom=0.05)
                plt.xticks(x_range, date_ticks, rotation=90, size=10)
                plt.yticks(y_range)
                plt.xlabel("OBS Time")
                plt.ylabel("Thresholds")
                plt.minorticks_on()
                plt.grid(True, which='major', c='gray', ls='-', lw=.75)
                plt.grid(True, which='minor', c='gray', ls='-', lw=.1)

            plt.tight_layout()
            return fig, ax

        else:
            fig, ax = plt.subplots(n, 1, figsize=(10, 5*n))
            for idx in range(n):
                bounds = np.round(np.linspace(0.0, 1.0, 21), 2)
                norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                im = ax[idx].pcolormesh(kinds[idx], norm=norm, cmap='jet')
                ax[idx].invert_yaxis()
                ax[idx].set_facecolor((0.8, 0.8, 0.8))
                ax[idx].title.set_text("{}\n{} to {}".format(
                    title[idx], str(start), str(end)))
                divider = make_axes_locatable(ax[idx])
                cax = divider.append_axes("right", size="2.5%", pad=0.05)
                plt.colorbar(im, ax=ax[idx], cax=cax, ticks=bounds)
                plt.sca(ax[idx])
                plt.subplots_adjust(bottom=0.05)
                plt.xticks(x_range, date_ticks, rotation=90, size=10)
                plt.yticks(y_range)
                plt.xlabel("OBS Time")
                plt.ylabel("Thresholds")
                plt.minorticks_on()
                plt.grid(True, which='major', c='gray', ls='-', lw=.75)
                plt.grid(True, which='minor', c='gray', ls='-', lw=.1)

            plt.tight_layout()
            return fig, ax

    def confusion_matrix(self):
        csi = np.zeros((self.threshold, self.pred.shape[0])) #shape[0] = 133(時間點)
        # far = np.zeros((self.threshold, self.pred.shape[0]), dtype=np.float16)
        # pod = np.zeros((self.threshold, self.pred.shape[0]), dtype=np.float16)
        # acc = np.zeros((self.threshold, self.pred.shape[0]), dtype=np.float16)
        
        for time in range(self.pred.shape[0]):#筆數(144,1)
            # print("--time=",time)
            for th in range(self.threshold):#60
                a = 0
                b = 0
                c = 0
                d = 0
                for idx in range(self.pred.shape[1]): #shape[1] = 121
                    if self.pred[time][idx] >= th and self.target[time][idx] >= th:
                        a += 1
                    elif self.pred[time][idx] >= th and self.target[time][idx] < th:
                        b += 1
                    elif self.pred[time][idx] < th and self.target[time][idx] >= th:
                        c += 1
                    else:
                        d += 1
                csi[th, time] = self.__CSI__(a, b, c)#csi[0~60,0] ~ csi[0~60,143]
                # print("csi[",th,",",time,"]=",csi[th, time])#(60, 99225)
                # far[th, time] = self.__FAR__(a, b)
                # pod[th, time] = self.__POD__(a, c)
                # acc[th, time] = self.__ACC__(a, b, c, d)
        #csi[60,144]
        # return csi, far, pod, acc
        # return acc
        print("csi1=")
        print(csi)
        return csi#(60, 99225)
    
    def visualize_csi(self, title=None):
        kinds = [self.csi]
        title = ['CSI'+title]
        return self._visualize(kinds=kinds, title=title)

    def visualize_far(self):
        kinds = [self.far]
        title = ['FAR']
        return self._visualize(kinds=kinds, title=title)

    def visualize_pod(self):
        kinds = [self.pod]
        title = ['POD']
        return self._visualize(kinds=kinds, title=title)

    def visualize_acc(self):
        kinds = [self.acc]
        title = ['ACC']
        return self._visualize(kinds=kinds, title=title)

    def visualize_all(self):
        kinds = [self.csi, self.far, self.pod, self.acc]
        title = ['CSI', 'FAR', 'POD', 'ACC']
        return self._visualize(kinds=kinds, title=title)

    def compare_csi(self, data):
        data = self.csi - data
        start = self.datetime[0].strftime("%Y%m%d-%H%M")
        end = self.datetime[-1].strftime("%Y%m%d-%H%M")
        date_ticks = [self.datetime[i].strftime("%H:%M") for i in range(0, len(self.datetime), 6)]
        x_range = [i for i in range(0, len(self.datetime), 6)]
        y_range = [i for i in range(1, self.threshold, 10)]

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        bounds = np.round(np.linspace(-1.0, 1.0, 21), 1)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        im = ax.pcolormesh(data, norm=norm, cmap='seismic')
        ax.set_facecolor((0.8, 0.8, 0.8))
        ax.title.set_text("{}\n{} to {}".format("DIFF with CSI", str(start), str(end)))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)
        plt.colorbar(im, ax=ax, cax=cax, ticks=bounds)
        ax.invert_yaxis()
        plt.sca(ax)
        plt.subplots_adjust(bottom=0.05)
        plt.xticks(x_range, date_ticks, rotation=90, size=10)
        plt.yticks(y_range)
        plt.xlabel("OBS Time")
        plt.ylabel("Thresholds")
        plt.minorticks_on()
        plt.grid(True, which='major', c='gray', ls='-', lw=.75)
        plt.grid(True, which='minor', c='gray', ls='-', lw=.1)
        plt.tight_layout()

        return fig, ax
