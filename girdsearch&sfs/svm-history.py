#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:39:32 2019

@author: shuojiawang
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, accuracy_score, auc, precision_recall_fscore_support
from sklearn import ensemble
from sklearn import tree
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


df = pd.read_csv('/Users/shuojiawang/Documents/ppdmodel/modelfilenew/wcmmodelwithhistory.csv', delimiter=',', header=0)

y = df['ppd']
#X = df.drop(columns=['ppd'])
X = df.drop(columns=['person_id', 'ppd','condition_start_date'])
#print(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)


clfsvm = SVC(class_weight='balanced',kernel='linear')
gridsvm = GridSearchCV(clfsvm,
                      param_grid={'C':np.arange(0.01,100,10)},
                    scoring='roc_auc',
                    cv=5,
                    return_train_score=True)
gridsvm.fit(X_train, y_train)
scores = pd.DataFrame(gridsvm.cv_results_)
print(gridsvm.best_estimator_)


clfsvm = SVC(C=0.01, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=True, random_state=None,
    shrinking=True, tol=0.001, verbose=False)

clfsvm.fit(X_train, y_train)
y_pred_c = clfsvm.predict(X_test)
y_pred_proba_csvm = clfsvm.predict_proba(X_test)[:, 1]
confmat_test_c = confusion_matrix(y_true=y_test, y_pred=y_pred_c)

print('confmat_test:\n', confmat_test_c)
print('the acc is:', accuracy_score(y_test, y_pred_c))
print('the classification_report:', classification_report(y_test, y_pred_c))
print('the auc of svm is:', roc_auc_score(y_test, y_pred_proba_csvm))




feature_names = X.columns.values.tolist()
print(feature_names)
print('ok2')

f_number = 50
sfs3 = SFS(clfsvm,
           k_features=f_number,
           forward=True,
           floating=False,
           scoring='roc_auc',
           cv=5)

print('ok3')
result3 = sfs3.fit(X_train, y_train, custom_feature_names=feature_names)
#print(X)
result3.subsets_
result3.k_score_

selection_res = pd.DataFrame.from_dict(sfs3.get_metric_dict()).T
# print(selection_res)
selection_res.to_csv("/Users/shuojiawang/Documents/ppdmodel/result1907/selection_log_withouthistorysvm.csv", sep='\t')

selected_feature_idx = result3.k_feature_idx_
#print(type(selected_feature_idx))
selected_feature = list(selected_feature_idx)
feature_name = []
for i in selected_feature:
    feature_name.append(feature_names[i])
print(feature_name)


fig = plot_sfs(sfs3.get_metric_dict(), kind='std_err')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.xlabel("Feature number")
plt.ylabel("AUC")
plt.grid()
#plt.savefig("Users/bu/Desktop/feature_selection.png", dpi=600)
plt.show()
#plt.clf()


