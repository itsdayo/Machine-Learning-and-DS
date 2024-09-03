#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing

@author: dayo
"""

import pandas as pd
# import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

Data_Set1 = pd.read_csv("Data_Set.csv")

Data_Set2 = pd.read_csv("Data_Set.csv", header=2)

Data_Set3 = Data_Set2.rename(columns = {"Temperature":"Temp"})

Data_Set4 = Data_Set3.drop("No. Occupants", axis=1)

# Data_Set3.drop("No. Occupants", axis=1, inplace= True)

Data_Set5 = Data_Set4.drop(2, axis=0)
Data_Set6 = Data_Set5.reset_index(drop= True)

print(Data_Set6.describe())

Min_item = Data_Set6['E_Heat'].min()

print(Data_Set6["E_Heat"][Data_Set6["E_Heat"]==Min_item])

# Data_Set6['E_Heat'].replace(-4,21, inplace=True)

# Data_Set6.replace({'E_Heat': 28}, inplace=True)

Data_Set6['E_Heat'] = Data_Set6['E_Heat'] .replace(-4,21)
Data_Set6['Price'] = Data_Set6['Price'] .replace("!",np.nan)

print(Data_Set6.cov())



# sn.heatmap(Data_Set6.corr())

Data_Set6 = Data_Set6.apply(pd.to_numeric)

print(Data_Set6.isnull())

print(Data_Set6.info())

Data_Set7 = Data_Set6.fillna(method ='ffill')

# print(Data_Set7)


'''
from sklearn.impute import SimpleImputer

M_var = SimpleImputer(missing_values=np.nan, strategy='mean')

M_var.fit(Data_Set6)

Data_Set8 = M_var.transform(Data_Set6)
'''


# plt.boxplot(Data_Set8)
# plt.show()

"""

Outlier Detection

"""

print(Data_Set7['E_Plug'].quantile(0.25))
print(Data_Set7['E_Plug'].quantile(0.75))

"""
Q1 = 21.25
Q3 = 33.75

IQR = 34.5 -20.5 = 14

Mild Outlier

Lower Bound = Q1 - 1.5*IQR = 21.25 -1.5*12.5 = 2.5
Upper Bound = Q3 +1.5*IQR = 33.75 +1.5*12.5 = 52.5

Extreme Outlier

Upper Bound = Q3 +3IQR = 33.75 +3*12.5 = 71.25


"""

Data_Set7['E_Plug'].replace(120,42, inplace=True)


New_Col =pd.read_csv("Data_New1.csv")

Data_Set9 = pd.concat([Data_Set7, New_Col], axis=1)


"""
Dummy Variables

"""

Data_Set11 = pd.get_dummies(Data_Set9)

print(Data_Set11.info())


"""
Normalization

"""

from sklearn.preprocessing import minmax_scale, normalize

# First Method: Min Max Scale

Data_Set12 = minmax_scale(Data_Set11, feature_range=(0,1))


print(Data_Set12)

Data_Set13 = normalize(Data_Set11, norm='l2' , axis=0)


Data_Set13 = pd.DataFrame(Data_Set13,columns=['Time','E_Plug','E_Heat','Price','Temp','OffPeak','Peak'])
print(Data_Set13)



