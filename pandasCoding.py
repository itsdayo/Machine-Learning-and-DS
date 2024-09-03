#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

pandas

@author: dayo
"""

import pandas as pd

Age = pd.Series([10,20,30,40],index=['age1','age2','age3','age4'])
Age.age3

Filtered_Age = Age[Age>20]

index = Age.index
values =Age.values

Age.index = ['A1', "A2", "A3","A4"]

import numpy as np

DF = np.array([[20,10,0],[25,8,10],[25,5,3],[30,9,7]])

Data_Set = pd.DataFrame(DF)
Data_Set = pd.DataFrame(DF, index=['S1', "S2",'S3','S4'])

Data_Set = pd.DataFrame(DF, index=['S1', "S2",'S3','S4'],columns=['Age', "Grade1",'Grade2'])
Data_Set['Grade3'] = [9,6,7,10]

S2 =Data_Set.loc['S2']


dataset_row= Data_Set.iloc[:,3]
bettyGrade = Data_Set.iloc[1][3]

Data_Set = Data_Set.replace(10,12)
Data_Set = Data_Set.replace({12:10, 9:30})
Data_Set =  Data_Set.sort_values("Grade1", ascending= True)

Data = pd.read_csv('Data_set.csv')
