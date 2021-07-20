# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 01:17:03 2019

@author: cindy
"""
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap

shape_path = 'visualize/mapdata201805310314/COUNTY_MOI_1070516'


m = Basemap(projection='merc', resolution='i', fix_aspect=True,
            llcrnrlon=119.0, llcrnrlat=21.8,
            urcrnrlon=122.05, urcrnrlat=25.4,
            lat_ts=20)

m.drawparallels(np.arange(18, 29.0125), fontsize=10)
m.drawmeridians(np.arange(115, 126.5125), fontsize=10)

m.readshapefile(shape_path , linewidth=0.25 , drawbounds=True, name='Taiwan')

m.drawcoastlines(linewidth=1)

x1, y1 = m(121.4338, 24.9994)
m.plot(x1, y1, marker = 'D', color = 'b')
plt.text(x1, y1, 'Banquiao')
#
#x2, y2 = m(121.5066, 25.0394)
#m.plot(x2, y2, marker = '.', color = 'r')
##plt.text(x2, y2, 'Taipei')
#
#x3, y3 = m(121.7322, 25.1351)
#m.plot(x3, y3, marker = '.', color = 'r')
##plt.text(x3, y3, 'Keelung')
#
x4, y4 = m(121.6051, 23.9769)
m.plot(x4, y4, marker = 'D', color = 'b')
plt.text(x4, y4, 'Hualien')
#
#x5, y5 = m(121.0393, 25.0085)
#m.plot(x5, y5, marker = '.', color = 'r')
##plt.text(x5, y5, 'Xinwu')
#
#x6, y6 = m(120.4248, 23.4977)
#m.plot(x6, y6, marker = '.', color = 'r')
##plt.text(x6, y6, 'Chiayi')
#
# x7, y7 = m(120.8957, 22.3574)
# m.plot(x7, y7, marker = '.', color = 'b')
# plt.text(x7, y7, 'Dawu')
#
# x8, y8 = m(120.7383, 22.0057)
# m.plot(x8, y8, marker = 'D', color = 'b')
# plt.text(x8, y8, 'Hengchun')
#
# x9, y9 = m(121.3653, 23.0993)
# m.plot(x9, y9, marker = '.', color = 'r')
# plt.text(x9, y9, 'Chenggong')
#
x10, y10 = m(120.8999, 23.8831)
m.plot(x10, y10, marker = '.', color = 'r')
plt.text(x10, y10, 'SunMoonLake')

x11, y11 = m(121.1465, 22.754)
m.plot(x11, y11, marker = 'D', color = 'b')
plt.text(x11, y11, 'Taitung')
#
x12, y12 = m(120.5152, 24.2578)
m.plot(x12, y12, marker = 'D', color = 'b')
plt.text(x12, y12, 'Wuqi')
#
# x13, y13 = m(120.5065, 23.8792)
# m.plot(x13, y13, marker = '.', color = 'r')
# plt.text(x13, y13, 'Beidou')

x14, y14 = m(120.2955, 23.6927)
m.plot(x14, y14, marker = '.', color = 'r')
plt.text(x14, y14, 'Baozhong')

x15, y15 = m(120.532, 22.5361)
m.plot(x15, y15, marker = '.', color = 'r')
plt.text(x15, y15, 'Chaojhou')

#x16, y16 = m(120.5748, 22.4698)
#m.plot(x16, y16, marker = '.', color = 'r')
##plt.text(x16, y16, 'Xinpi')
#
#x17, y17 = m(121.7224, 24.7443)
#m.plot(x17, y17, marker = '.', color = 'r')
##plt.text(x17, y17, 'Yuanshan')
#
#x18, y18 = m(121.3316, 23.3233)
#m.plot(x18, y18, marker = '.', color = 'r')
##plt.text(x18, y18, 'Yuli')
#
#x19, y19 = m(121.0186, 24.2825)
#m.plot(x19, y19, marker = '.', color = 'r')
##plt.text(x19, y19, 'Xueling')

x20, y20 = m(120.6964, 22.765)
m.plot(x20, y20, marker = '.', color = 'r')
plt.text(x20, y20, 'Shangdewun')

plt.xlabel('lon' , fontsize=12 , x=1)
plt.ylabel('lat' , fontsize=12 , y=1)

plt.savefig('Taiwan_NoCityName.png', dpi=200)

plt.show()
