# -*- coding: utf-8 -*-
"""
Numpy 

"""

import numpy as np

NumP_Array =np.array([[1,2,3],[4,5,6]])

NP1  = np.array([[1,3],[4,5]])
NP2 = np.array([[3,4],[5,7]])

#matrix multiplication

NMNP =NP1@NP2

NMP3 = np.dot(NP1, NP2)

# element by element multiplication 
NMP2=NP1*NP2

NMP4 = np.multiply(NP1,NP2)



Sum1 = NP1 + NP2

Diff1 = NP1- NP2
Diff2 = np.subtract(NP1,NP2)

# division 
D = np.divide([12,14,16],5)

mathsqrt = np.math.sqrt(16)

normal_distribution = np.random.standard_normal((3,4))
uniform_distribution = np.random.uniform(1,12, (3,4))

Random_Arr = np.random.randint(1,50,(2,5))

filter_Arr = np.logical_and(Random_Arr>30,Random_Arr<50)
F_Random_Arr = Random_Arr[filter_Arr]


Data_N = np.array([1,3,4,5,7,9])

Mean_N = np.mean(Data_N)
Median_N =np.median(Data_N)

Varience_N = np.var(Data_N)

Standard_deviation_N = np.std(Data_N) 

NumP_arr = np.array([[1,2,3]])

# goes to each row and calculates varience

Var_Nump = np.var(NumP_arr, axis=1)

# goes to each column and calculates varience

Var_Nump2 = np.var(NumP_arr, axis=0)



