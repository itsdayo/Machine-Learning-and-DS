"""

SVR Hyper Paramter Tuning

"""



from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


cali_housing = fetch_california_housing()
x = cali_housing.data
y = cali_housing.target




from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# parameters = {'kernel':['rbf','linear'], "gamma":[1,0.1,0,0.01]}

# grid =GridSearchCV(SVR(),parameters,refit=True,verbose=2, scoring='neg_mean_squared_error')

# grid.fit(x,y)

# best_parameters = grid.best_params_

# print("best parameters",'\n',best_parameters)


"""

k-NN Hyper Parameter Tuning

"""


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

x =iris.data
y = iris.target


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, train_size=0.75, random_state=88,shuffle=True,stratify=y )



from sklearn.neighbors import KNeighborsClassifier

kNN_accuracy_train = []
kNN_accuracy_test = []

for k in range(1,50):
    kNN = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=1)
    kNN.fit(x_train, y_train)
    kNN_accuracy_train.append(kNN.score(x_train,y_train))
    kNN_accuracy_test.append(kNN.score(x_test,y_test))


plt.plot(np.arange(1,50), kNN_accuracy_train, label='train')
plt.plot(np.arange(1,50), kNN_accuracy_test, label='test')
plt.xlabel("K")
plt.ylabel("Score")
plt.legend(loc='best')
plt.show()




