#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization with Matplotlib
@author: dayo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Year =[2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

Temp = [0.72, 0.62,0.65,0.68,0.75,0.90,1.02,0.93,0.85,0.99,1.02]

plt.plot(Year,Temp)

plt.xlabel("Year")
plt.ylabel("Temperature")
plt.title("Global Warming",{'fontsize':12,'horizontalalignment':'center'})
plt.show()
plt.close()

Month= ['Jan', 'Feb','Mar','Apr','May','June','July','Aug',"Sep",'Oct','Nov','Dec']

Customer1 = [12,13,9,8,7,8,8,7,6,5,8,10]
Customer2 = [14,16,11,7,6,6,7,6,5,8,9,12]

plt.plot(Month, Customer1, color='limegreen', label='Customer 1', marker='.')
plt.plot(Month, Customer2, color='silver', label= "Customer 2", marker='.')
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Building Consumption")
plt.legend(loc='center')
plt.show()


# row column number or graphs
plt.subplot(1,2,1)

plt.plot(Month, Customer1, color='limegreen', label='Customer 1', marker='.')
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Building Consumption of Customer 1")


plt.subplot(1,2,2)

plt.plot(Month, Customer2, color='silver', label= "Customer 2", marker='.')
plt.xlabel("Month")

plt.title("Building Consumption of Customer 2")
plt.show()


plt.scatter(Month, Customer1, color='blue', label='Customer1')
plt.scatter(Month, Customer2, color='red', label='Customer2')
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Scatter plot of Building Consumption")
plt.grid()
plt.legend(loc='best')
plt.show()

plt.hist(Customer1,bins=40, color='green')
plt.ylabel("Electricity Consumption")
plt.title("Histogram")
plt.show()

plt.bar(Month, Customer1, width=0.8, color = 'b')
plt.show()


plt.bar(Month, Customer1, color='blue', label='Customer1')
plt.bar(Month, Customer2, color='red', label='Customer2')
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Bar Chart of Building Consumption")
plt.legend(loc='best')
plt.show()

bar_width = 0.4
Month_b= np.arange(12)

plt.bar(Month_b, Customer1, bar_width,color='blue' ,label ="Customer1")
plt.bar(Month_b+bar_width,Customer2 ,bar_width, color='red', label='Customer2')
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Bar Chart of Building Consumption")
plt.legend(loc='best')


plt.xticks(Month_b+(bar_width)/12, ('Jan', 'Feb','Mar','Apr','May','June','July','Aug',"Sep",'Oct','Nov','Dec'))
plt.show()

plt.boxplot(Customer1, notch=True, vert=False)

plt.boxplot([Customer1,Customer2], patch_artist=True ,
            boxprops = dict(facecolor='red',color='red'),
            whiskerprops=dict(color='green'),
            capprops=dict(color='blue'),
            medianprops=dict(color='yellow'),notch=True, vert=False)
plt.show()


