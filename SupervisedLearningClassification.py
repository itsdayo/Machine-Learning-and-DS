import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()

iris.feature_names

data_iris = iris.data

data_iris = pd.DataFrame(data_iris, columns=iris.feature_names)


data_iris['label']= iris.target

print(data_iris)


plt.scatter(x=data_iris.iloc[:,2], y=data_iris.iloc[:,3], c=iris.target)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")

plt.show()

x =data_iris.iloc[:, 0:4]
y = data_iris.iloc[:,4]

'''

k-NNN Classifier

'''

from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors = 6, metric='minkowski', p=1)

kNN.fit(x,y)

x_new = np.array([[5.6,3.4,1.4,0.1]])

print(kNN.predict(x_new))

x_new2 = np.array([[7.5,4,5.5,2]])
print(kNN.predict(x_new2))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, train_size=0.8, 
                                                    random_state=88, shuffle=True,
                                                    stratify=y) 
print(y_test)

kNN2 = KNeighborsClassifier(n_neighbors = 6, metric='minkowski', p=1)

kNN2.fit(x_train,y_train)

x_new = np.array([[5.6,3.4,1.4,0.1]])

print(kNN2.predict(x_test))

from sklearn.metrics import  accuracy_score
print(accuracy_score(y_test, kNN2.predict(x_test)))


"""
Decision Tree

"""

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

predicted_dt_types = dt.predict(x_test)

print(predicted_dt_types)
print(accuracy_score(y_test, predicted_dt_types))

from sklearn.model_selection import cross_val_score


# cross validation
scores_dt =cross_val_score(dt,x,y,cv=10)

print(scores_dt)



'''
Naive Bayes Classifier

'''

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)

predicted_types_nb = nb.predict(x_test)


accuracy_score_nb = accuracy_score(y_test, predicted_types_nb)

scores_nb= cross_val_score(nb,x,y,cv=10)

print("naive bayer", predicted_types_nb)
print("naive bayer accuracy", accuracy_score_nb)
print("naive bayer cross val score", scores_nb)


'''

Logisitic Regression

'''

from sklearn.datasets import load_breast_cancer

data_c =load_breast_cancer()

x = data_c.data
y = data_c.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,train_size=0.7, random_state=88)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, y_train)

predicted_types_lr = lr.predict(x_test)

accuracy_score_lr = accuracy_score(y_test, predicted_types_lr)

scores_lr= cross_val_score(nb,x,y,cv=10)

print("logisitic regression", predicted_types_lr)
print("logisitic regression accuracy", accuracy_score_lr)
print("logisitic regression cross val score", scores_lr)

plt.show()
"""

Evaluation Metrics

"""

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

conf_matrix = confusion_matrix(y_test, predicted_types_lr)

class_report = classification_report(y_test, predicted_types_lr)

print("confusion matrix", conf_matrix)
print("clasificationreport ","\n" ,class_report)

y_prob =  lr.predict_proba(x_test)

print("predict probability logistic regression","\n" ,y_prob)


y_prob = y_prob[:,1]

FPR, TPR , Thresholds = roc_curve(y_test,y_prob)

plt.plot(FPR, TPR)

plt.xlabel("FPR")
plt.xlabel("TPR")

auc_score =  roc_auc_score(y_test, y_prob)
print("auc score","\n" ,auc_score)

plt.show()


