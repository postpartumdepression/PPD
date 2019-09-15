#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:29:33 2019

@author: shuojiawang
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs5
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, accuracy_score, auc, precision_recall_fscore_support
from sklearn import ensemble
from sklearn import tree
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


df = pd.read_csv('/Users/shuojiawang/Documents/ppdmodel/modelfilenew/wcmmodelwithhistory.csv', delimiter=',')

y = df['ppd']
X = df.drop(columns=['person_id', 'ppd','condition_start_date'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)




decisiontree = tree.DecisionTreeClassifier(class_weight='balanced')
griddt = {'min_samples_split' : range(10,500,20),
          'max_depth': range(1,30,1)}
#griddt = {'max_depth': np.arange(3, 10)}
griddt_cv = GridSearchCV(decisiontree, griddt, refit = True, scoring = 'roc_auc',cv=5)
griddt_cv.fit(X_train, y_train)
print("tuned decision tree:(best parameters) ",griddt_cv.best_estimator_)

feature_names = X.columns.values.tolist()
print(feature_names)

clfDecisionTree = tree.DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=13,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=290,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')

clfDecisionTree.fit(X_train, y_train)
y_pred_c = clfDecisionTree.predict(X_test)
y_pred_proba_cdt = clfDecisionTree.predict_proba(X_test)[:, 1]
confmat_test_c = confusion_matrix(y_true=y_test, y_pred=y_pred_c)

print('confmat_test:\n', confmat_test_c)
print('the acc is:', accuracy_score(y_test, y_pred_c))
print('the classification_report:', classification_report(y_test, y_pred_c))
print('the auc of decision tree is:', roc_auc_score(y_test, y_pred_proba_cdt))

#confmat_test:
# [[2930  379]
# [  16   52]]
#the acc is: 0.883032277169085
#the classification_report:               precision    recall  f1-score   support
#
#         0.0       0.99      0.89      0.94      3309
#         1.0       0.12      0.76      0.21        68
#
#    accuracy                           0.88      3377
#   macro avg       0.56      0.83      0.57      3377
#weighted avg       0.98      0.88      0.92      3377
#
#the auc of decision tree is: 0.8703024727570086

f_number = 50
sfs5 = sfs(clfDecisionTree,
           k_features=f_number,
           forward=True,
           floating=False,
           scoring='roc_auc',
           cv=5)

print('ok3')
result5 = sfs5.fit(X_train, y_train, custom_feature_names=feature_names)
#print(X)
result5.subsets_
result5.k_score_


selection_res = pd.DataFrame.from_dict(sfs5.get_metric_dict()).T
# print(selection_res)
selection_res.to_csv("/Users/shuojiawang/Documents/ppdmodel/result1907/selection_log_withhistorydt.csv", sep='\t')

selected_feature_idx = result5.k_feature_idx_
#print(type(selected_feature_idx))
selected_feature = list(selected_feature_idx)
feature_name = []
for i in selected_feature:
    feature_name.append(feature_names[i])
print(feature_name)


fig = plot_sfs5(sfs5.get_metric_dict(), kind='std_err')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.xlabel("Feature number")
plt.ylabel("AUC")
plt.grid()
#plt.savefig("Users/bu/Desktop/feature_selection.png", dpi=600)
plt.show()
#plt.clf()

