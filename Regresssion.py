"""

Regression

"""


from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


cali_housing = fetch_california_housing()
x = cali_housing.data
y = cali_housing.target

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, train_size=0.75, random_state=88)

sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

y_train = y_train.reshape(-1,1)
y_train = sc.fit_transform(y_train)


"""

Multiple Linear Regression

"""


from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

linear_reg.fit(x_train, y_train)

predicted_values_mlr = linear_reg.predict(x_test)

predicted_values_mlr = sc.inverse_transform(predicted_values_mlr)

print(predicted_values_mlr, y_test)


"""

Evaluation Metrics

"""


from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import math

MAE =mean_absolute_error(y_test, predicted_values_mlr)
MSE = mean_squared_error(y_test, predicted_values_mlr)
RMSE =math.sqrt(MSE)

R2 = r2_score(y_test,predicted_values_mlr)


print("mean absolute error", '\n', MAE)
print("mean square error", '\n', MSE)
print("root mean square error", '\n', RMSE)
print("r2", '\n', R2)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true -y_pred)/y_true))*100

MAPE = mean_absolute_percentage_error(y_test, predicted_values_mlr)

print("mean absolute percentage error", '\n', MAPE)


"""

Polynominal Linear Regression

"""

cali_housing_2 = fetch_california_housing()

x = cali_housing_2.data[:,5]

y = cali_housing_2.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, train_size=0.75, random_state=88)

from sklearn.preprocessing import PolynomialFeatures

poly_p = PolynomialFeatures(degree=2)

x_train = x_train.reshape(-1,1)
poly_x = poly_p.fit_transform(x_train)



poly_lr = linear_reg.fit(poly_x,y_train)

x_test = x_test.reshape(-1,1)
poly_xt = poly_p.fit_transform(x_test)

predicted_values_poly = poly_lr.predict(poly_xt)


"""

Random Forest

"""

from sklearn.ensemble import RandomForestRegressor

cali_housing = fetch_california_housing()
x = cali_housing.data
y = cali_housing.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, train_size=0.75, random_state=88)

sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

y_train = y_train.reshape(-1,1)
y_train = sc.fit_transform(y_train)

random_forest = RandomForestRegressor(n_estimators=10, max_depth= 20, random_state=33)
random_forest.fit(x_train, y_train)

predicted_values_rf = random_forest.predict(x_test)
predicted_values_rf  = predicted_values_rf.reshape(-1,1)

predicted_values_rf = sc.inverse_transform(predicted_values_rf)


print("random forest tree ytest","\n",y_test,"\n","values","\n",predicted_values_rf)


"""

Support Vector Regression

"""

from sklearn.svm import SVR

regressor_svr = SVR(kernel = 'rbf')

regressor_svr.fit(x_train, y_train)

predicted_values_svr = regressor_svr.predict(x_test)

predicted_values_svr = predicted_values_svr.reshape(-1,1)

predicted_values_svr = sc.inverse_transform(predicted_values_svr)

print("support vector regression values",'\n', predicted_values_svr)









