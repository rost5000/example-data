import matplotlib.pyplot as plt
from sklearn import datasets, svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rnd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

np.random.seed(0)

data_test = pd.read_csv('https://raw.githubusercontent.com/rost5000/example-data/master/hw1_test.csv')
data_train = pd.read_csv('https://raw.githubusercontent.com/rost5000/example-data/master/hw1_train.csv')

X_test = data_test.drop(["Target"], axis=1)
Y_test = data_test["Target"]

X_train = data_train.drop(["Target"], axis=1)
Y_train= data_train["Target"]

selector = SelectKBest(mutual_info_classif, 1)
selector.fit(X_train, Y_train)
 #Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X_train, Y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

#for f in range(X_train.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(X_train.shape[1]), importances[indices],
#       color="r", yerr=std[indices], align="center")
#plt.xticks(range(X_train.shape[1]), indices)
#plt.xlim([-1, X_train.shape[1]])
#plt.show()

values_indx = [indx for indx in indices]

values_str_indx = ["Time-{}".format(i) for i in values_indx[:50]]

data_class1 = data_train[data_train['Target'] == 1]
data_class2 = data_train[data_train['Target'] == -1]

X_class1 = data_class1[values_str_indx]
X_class2 = data_class2[values_str_indx]
np.array()

plt.figure()
plt.title("Feature importances")
tst = np.mean(X_class2.as_matrix().T)
plt.plot(np.mean(X_class2.as_matrix().T))
#plt.plot(np.mean(X_class1.as_matrix().T), 'r-')
#plt.legend("class2", "class1")
plt.show()