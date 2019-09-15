#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:09:37 2019

@author: bu
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
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
from sklearn.metrics import brier_score_loss

###########################################################################
df = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodelwithhistory_2.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visitexafter1ywcmnew_2.csv', sep=',')
visitexafter1y = pd.DataFrame(visitexafter1y,columns=['person_id','condition_start_date'])
df_2 = pd.merge(df,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df_2['ppd'].value_counts()

#feature_names = X.columns.values.tolist()#[2:]

#withouhistory
#clfRandomForest = ensemble.RandomForestClassifier(bootstrap=True, class_weight='balanced',
#                       criterion='gini', max_depth=8, max_features='auto',
#                       max_leaf_nodes=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=21,
#                       min_samples_split=140, min_weight_fraction_leaf=0.0,
#                       n_estimators=10, n_jobs=None, oob_score=False,
#                       random_state=None, verbose=0, warm_start=False)

clfRandomForest = ensemble.RandomForestClassifier(bootstrap=True, class_weight='balanced',
                                                  criterion='gini', max_depth=8,min_impurity_decrease=0.0,
                                                  random_state=12,
                                                  min_samples_leaf=21,min_samples_split=140,n_estimators=10)
                                                  
columnsrf88 = ['ppd','dbp2nd', 'asian', 'single', 'C1187.0', 'C1203.0', 'C1249.0', 'C1310.0', 'C1366.0', 'C1387.0', 'C1685.0', 'C1686.0', 'C1699.0', 'C1868.0', 'C19.0', 'C21.0', 'C2152.0', 'C2168.0', 'C2171.0', 'C2210.0', 'C2257.0', 'C2270.0', 'C2275.0', 'C2276.0', 'C2280.0', 'C2288.0', 'C2289.0', 'C2332.0', 'C2356.0', 'C2398.0', 'C2400.0', 'C2431.0', 'C2517.0', 'C2652.0', 'C2810.0', 'C2842.0', 'C2851.0', 'C2903.0', 'C2959.0', 'C3006.0', 'C3082.0', 'C3134.0', 'C3158.0', 'C3161.0', 'C3251.0', 'C3261.0', 'C3282.0', 'C345.0', 'C35.0', 'C3557.0', 'C3613.0', 'C372.0', 'C3734.0', 'C3796.0', 'C3805.0', 'C3866.0', 'C391.0', 'C397.0', 'C4001.0', 'C419.0', 'C4230.0', 'C4366.0', 'C4448.0', 'C4568.0', 'C4639.0', 'C4645.0', 'C4696.0', 'C4728.0', 'C4766.0', 'C4771.0', 'C4804.0', 'C4923.0', 'C497.0', 'C505.0', 'C56.0', 'C6.0', 'C624.0', 'C688.0', 'C689.0', 'C732.0', 'C853.0', 'C912.0', 'M104.0', 'M122.0', 'M24.0', 'M62.0', 'M67.0', 'M78.0', 'M98.0']


#withhistory
#clfRandomForest = ensemble.RandomForestClassifier(bootstrap=True, class_weight='balanced',
#                       criterion='gini', max_depth=17, max_features='auto',
#                       max_leaf_nodes=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=1,
#                       min_samples_split=120, min_weight_fraction_leaf=0.0,
#                       n_estimators=10, n_jobs=None, oob_score=False,
#                       random_state=None, verbose=0, warm_start=False)

#clfRandomForest = ensemble.RandomForestClassifier(bootstrap=True, class_weight='balanced',
#                                                  criterion='gini', max_depth=17, min_impurity_decrease=0.0,
#                                                  min_samples_leaf=1,
#                                                  min_samples_split=120, min_weight_fraction_leaf=0.0,
#                                                  n_estimators=10,verbose=0, warm_start=False)
#                                                  


columnsrf28 = ['ppd','dbp2nd', 'combine', 'C1209.0', 'C1253.0', 'C145.0', 'C1655.0', 'C1844.0', 'C209.0', 'C2168.0', 'C2471.0', 'C2517.0', 'C2644.0', 'C2906.0', 'C3174.0', 'C3267.0', 'C3447.0', 'C3624.0', 'C3626.0', 'C3734.0', 'C4126.0', 'C4179.0', 'C4468.0', 'C4764.0', 'C4774.0', 'M13.0', 'anxiety', 'mooddisorder', 'otherdisorder']

###########################################################################
columnsrf = columnsrf88 + columnsrf28
columnsrf = list(dict.fromkeys(columnsrf))

columnsrf2 = ['ppd',
 'dbp2nd',
 'asian',
 'single',
 #'C1187.0',
 #'C1203.0',
 #'C1249.0',
 #'C1310.0',
 #'C1366.0',
 'C1387.0',#Diarrhea
 'C1685.0',#Elderly primigravida
 'C1686.0',#Elderly primigravida with antenatal problem
 #'C1699.0',
 #'C1868.0',
 #'C19.0',
 #'C21.0',
 #'C2152.0',
 #'C2168.0',
 'C2171.0',#Headache
 #'C2210.0',
 #'C2257.0',
 #'C2270.0',
 #'C2275.0',
 #'C2276.0',
 'C2280.0',#High risk pregnancy
 'C2288.0',#History of recurrent miscarriage - not delivered
 #'C2289.0',
 #'C2332.0',
 #'C2356.0',
 'C2398.0',#Hypothyroidism
 #'C2400.0',
 #'C2431.0',
 #'C2517.0',
 #'C2652.0',#Irregular periods
 #'C2810.0',
 #'C2842.0',
 #'C2851.0',
 'C2903.0',#Depressed mood
 #'C2959.0',
 #'C3006.0',
 #'C3082.0',#Missed period
 #'C3134.0',
 #'C3158.0',
 #'C3161.0',
 #'C3251.0',
 #'C3261.0',
 #'C3282.0',
 #'C345.0',#Amenorrhea
 #'C35.0',
 #'C3557.0',
 #'C3613.0',
 #'C372.0',
 #'C3734.0',
 #'C3796.0',
 #'C3805.0',
 #'C3866.0',
 #'C391.0',
 #'C397.0',
 #'C4001.0',
 'C419.0',#Anxiety
 #'C4230.0',
 #'C4366.0',
 #'C4448.0',
 #'C4568.0',
 #'C4639.0',
 #'C4645.0',
 'C4696.0',#Uncomplicated asthma
 'C4728.0',#Urinary tract infectious disease
 #'C4766.0',
 #'C4771.0',
 #'C4804.0',
 #'C4923.0',
# 'C497.0',
 #'C505.0',
 #'C56.0',
 #'C6.0',
 #'C624.0',
 #'C688.0',
 'C689.0',#Blood coagulation disorder
 #'C732.0',
 #'C853.0',
 #'C912.0',
 #'M104.0',
 #'M122.0',
 'M24.0',#ANTIINFLAMMATORY AGENTS
 #'M62.0',
 #'M67.0',
 'M78.0',#IRON PREPARATIONS
 'M98.0',#OTHER PLAIN VITAMIN PREPARATIONS
 'other',
 #'C1209.0',
 'C1253.0',#Cough
 #'C145.0',
 #'C1655.0',
 #'C1844.0',
 #'C209.0',
 #'C2471.0',
 #'C2644.0',
 #'C2906.0',
 'C3174.0',#Nausea and vomiting
 #'C3267.0',#Normal birth
 #'C3447.0',
 #'C3624.0',
 'C3626.0',#Placental infarct
 'C4126.0',#Second degree perineal laceration
 #'C4179.0',
 #'C4468.0',
 #'C4764.0',
 #'C4774.0',
 #'M13.0',
 'anxiety',
 'mooddisorder',
 'otherdisorder']

#wcm
df3 = pd.DataFrame(df_2, columns=columnsrf2)
#df3.isnull().any()
#df3[np.isnan(df3)] = '0'
#df3.head(2)
y = df3['ppd']
X = df3.drop(columns=['ppd'])
X_train_sfsrf, X_test_sfsrf, y_trainrf, y_testrf = train_test_split(X, y, test_size=0.2, random_state=17)
print(X_test_sfsrf.shape)
print(X_train_sfsrf.shape)


clfRandomForest.fit(X_train_sfsrf, y_trainrf)
y_pred_crf = clfRandomForest.predict(X_test_sfsrf)
y_pred_proba_crf = clfRandomForest.predict_proba(X_test_sfsrf)[:, 1]
confmat_test_crf = confusion_matrix(y_true=y_testrf, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(y_testrf, y_pred_crf))
print('the classification_report:', classification_report(y_testrf, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(y_testrf, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(y_testrf, y_pred_proba_crf))

#without history

#confmat_test:
# [[2750  113]
# [  28  149]]
#the acc is: 0.9536184210526316
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.96      0.98      2863
#        1.0       0.57      0.84      0.68       177
#
#avg / total       0.97      0.95      0.96      3040
#
#the auc of random forest is: 0.9349858214389316
#the brier socre is 0.06860114778644015

import pickle 
with open('/Users/bu/Desktop/clfrandomforest_2.pickle', 'wb') as f:
    pickle.dump(clfRandomForest, f)
###########################################################################
#wcmmodel30
dfwcmmodel30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel30_2.csv', delimiter=',', header=0)
dfwcmmodel30 = pd.merge(dfwcmmodel30,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel30 = pd.DataFrame(dfwcmmodel30, columns=columnsrf2)
df2wcmmodel30[np.isnan(df2wcmmodel30)] = '0'
y30 = df2wcmmodel30['ppd']
X30 = df2wcmmodel30.drop(columns=['ppd'])

X_train30, X_test30, y_train30, y_test30 = train_test_split(X30, y30, test_size=0.2, random_state=17)

clfRandomForest.fit(X_train_sfsrf, y_trainrf)
y_pred_crf = clfRandomForest.predict(X30)
y_pred_proba_crf = clfRandomForest.predict_proba(X30)[:, 1]
confmat_test_crf = confusion_matrix(y_true=y30, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(y30, y_pred_crf))
print('the classification_report:', classification_report(y30, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(y30, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(y30, y_pred_proba_crf))

#without history
#confmat_test:
# [[13745   442]
# [  176   834]]
#the acc is: 0.9593340790945581
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98     14187
#        1.0       0.65      0.83      0.73      1010
#
#avg / total       0.97      0.96      0.96     15197
#
#the auc of random forest is: 0.9139903565319528
#the brier socre is 0.055759472883004896
###########################################################################
#wcmmodel24
dfwcmmodel24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel24_2.csv', delimiter=',', header=0)
dfwcmmodel24 = pd.merge(dfwcmmodel24,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel24 = pd.DataFrame(dfwcmmodel24, columns=columnsrf2)
df2wcmmodel24[np.isnan(df2wcmmodel24)] = '0'
y24 = df2wcmmodel24['ppd']
X24 = df2wcmmodel24.drop(columns=['ppd'])
X_train24, X_test24, y_train24, y_test24 = train_test_split(X24, y24, test_size=0.2, random_state=17)

clfRandomForest.fit(X_train_sfsrf, y_trainrf)
y_pred_crf = clfRandomForest.predict(X_test24)
y_pred_proba_crf = clfRandomForest.predict_proba(X_test24)[:, 1]
confmat_test_crf = confusion_matrix(y_true=y_test24, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(y_test24, y_pred_crf))
print('the classification_report:', classification_report(y_test24, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(y_test24, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(y_test24, y_pred_proba_crf))

#without history
#confmat_test:
# [[2769   94]
# [  35  142]]
#the acc is: 0.9575657894736842
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2863
#        1.0       0.60      0.80      0.69       177
#
#avg / total       0.97      0.96      0.96      3040
#
#the auc of random forest is: 0.9028477496837699
#the brier socre is 0.05671285087759234
###########################################################################
#wcmmodel18
dfwcmmodel18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel18_2.csv', delimiter=',', header=0)
dfwcmmodel18 = pd.merge(dfwcmmodel18,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel18 = pd.DataFrame(dfwcmmodel18, columns=columnsrf2)
df2wcmmodel18[np.isnan(df2wcmmodel18)] = '0'
y18 = df2wcmmodel18['ppd']
X18 = df2wcmmodel18.drop(columns=['ppd'])
X_train18, X_test18, y_train18, y_test18 = train_test_split(X18, y18, test_size=0.2, random_state=17)

clfRandomForest.fit(X_train_sfsrf, y_trainrf)
y_pred_crf = clfRandomForest.predict(X_test18)
y_pred_proba_crf = clfRandomForest.predict_proba(X_test18)[:, 1]
confmat_test_crf = confusion_matrix(y_true=y_test18, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(y_test18, y_pred_crf))
print('the classification_report:', classification_report(y_test18, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(y_test18, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(y_test18, y_pred_proba_crf))

##withouthistory
#confmat_test:
# [[2770   93]
# [  35  142]]
#the acc is: 0.9578947368421052
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2863
#        1.0       0.60      0.80      0.69       177
#
#avg / total       0.97      0.96      0.96      3040
#
#the auc of random forest is: 0.8969928031715774
#the brier socre is 0.05649286060070705
###########################################################################
#wcmmodel12
dfwcmmodel12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel12_2.csv', delimiter=',', header=0)
dfwcmmodel12 = pd.merge(dfwcmmodel12,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel12 = pd.DataFrame(dfwcmmodel12, columns=columnsrf2)
df2wcmmodel12[np.isnan(df2wcmmodel12)] = '0'
y12 = df2wcmmodel12['ppd']
X12 = df2wcmmodel12.drop(columns=['ppd'])
X_train12, X_test12, y_train12, y_test12 = train_test_split(X12, y12, test_size=0.2, random_state=17)

clfRandomForest.fit(X_train_sfsrf, y_trainrf)
y_pred_crf = clfRandomForest.predict(X_test12)
y_pred_proba_crf = clfRandomForest.predict_proba(X_test12)[:, 1]
confmat_test_crf = confusion_matrix(y_true=y_test12, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(y_test12, y_pred_crf))
print('the classification_report:', classification_report(y_test12, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(y_test12, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(y_test12, y_pred_proba_crf))

#without history
#confmat_test:
# [[2770   93]
# [  35  142]]
#the acc is: 0.9578947368421052
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2863
#        1.0       0.60      0.80      0.69       177
#
#avg / total       0.97      0.96      0.96      3040
#
#the auc of random forest is: 0.8968576283026575
#the brier socre is 0.054086674399064845
###########################################################################
#cdrn
dfcdrn = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew_2.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew_2.csv', delimiter=',', header=0)
dfcdrn = pd.merge(dfcdrn,visitexafter1y,on = ['person_id','condition_start_date'], how='inner')
dfcdrn['ppd'].value_counts()
df2cdrn = pd.DataFrame(dfcdrn, columns=columnsrf2)
df2cdrn[np.isnan(df2cdrn)] = '0'
ycdrn = df2cdrn['ppd']
Xcdrn = df2cdrn.drop(columns=['ppd'])

clfRandomForest.fit(X_train_sfsrf, y_trainrf)
y_pred_crf = clfRandomForest.predict(Xcdrn)
y_pred_proba_crf = clfRandomForest.predict_proba(Xcdrn)[:, 1]
confmat_test_crf = confusion_matrix(y_true=ycdrn, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(ycdrn, y_pred_crf))
print('the classification_report:', classification_report(ycdrn, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(ycdrn, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(ycdrn, y_pred_proba_crf))

#without history

#confmat_test:
# [[41573  8886]
# [  467  3046]]
#the acc is: 0.8267064403764915
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.82      0.90     50459
#        1.0       0.26      0.87      0.39      3513
#
#avg / total       0.94      0.83      0.87     53972
#
#the auc of random forest is: 0.8603790051054632
#the brier socre is 0.15403707087446805

###########################################################################
#cdrn30
dfcdrn30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew30_2.csv', delimiter=',', header=0)
dfcdrn30 = pd.merge(dfcdrn30,visitexafter1y,on = ['person_id','condition_start_date'], how='inner')
df2cdrn30 = pd.DataFrame(dfcdrn30, columns=columnsrf2)

df2cdrn30[np.isnan(df2cdrn30)] = '0'
ycdrn30 = df2cdrn30['ppd']
Xcdrn30 = df2cdrn30.drop(columns=['ppd'])

clfRandomForest.fit(X_train_sfsrf, y_trainrf)
y_pred_crf = clfRandomForest.predict(Xcdrn30)
y_pred_proba_crf = clfRandomForest.predict_proba(Xcdrn30)[:, 1]
confmat_test_crf = confusion_matrix(y_true=ycdrn30, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(ycdrn30, y_pred_crf))
print('the classification_report:', classification_report(ycdrn30, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(ycdrn30, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(ycdrn30, y_pred_proba_crf))
##without history
#confmat_test:
# [[42376  8083]
# [  911  2602]]
#the acc is: 0.8333580375009264
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.84      0.90     50459
#        1.0       0.24      0.74      0.37      3513
#
#avg / total       0.93      0.83      0.87     53972
#
#the auc of random forest is: 0.806721831305665
#the brier socre is 0.1475575075983434
###########################################################################
#cdrn24
dfcdrn24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew24_2.csv', delimiter=',', header=0)
dfcdrn24 = pd.merge(dfcdrn24,visitexafter1y,on = ['person_id','condition_start_date'], how='inner')
df2cdrn24 = pd.DataFrame(dfcdrn24, columns=columnsrf2)

df2cdrn24[np.isnan(df2cdrn24)] = '0'
ycdrn24 = df2cdrn24['ppd']
Xcdrn24 = df2cdrn24.drop(columns=['ppd'])

clfRandomForest.fit(X_train_sfsrf, y_trainrf)
y_pred_crf = clfRandomForest.predict(Xcdrn24)
y_pred_proba_crf = clfRandomForest.predict_proba(Xcdrn24)[:, 1]
confmat_test_crf = confusion_matrix(y_true=ycdrn24, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(ycdrn24, y_pred_crf))
print('the classification_report:', classification_report(ycdrn24, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(ycdrn24, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(ycdrn24, y_pred_proba_crf))

#withouthistory
#confmat_test:
# [[42512  7947]
# [  936  2577]]
#the acc is: 0.8354146594530497
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.84      0.91     50459
#        1.0       0.24      0.73      0.37      3513
#
#avg / total       0.93      0.84      0.87     53972
#
#the auc of random forest is: 0.8002850146500557
#the brier socre is 0.1463134697855688
###########################################################################
#cdrn18
dfcdrn18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew18_2.csv', delimiter=',', header=0)
dfcdrn18 = pd.merge(dfcdrn18,visitexafter1y,on = ['person_id','condition_start_date'], how='inner')
df2cdrn18 = pd.DataFrame(dfcdrn18, columns=columnsrf2)

df2cdrn18[np.isnan(df2cdrn18)] = '0'
ycdrn18 = df2cdrn18['ppd']
Xcdrn18 = df2cdrn18.drop(columns=['ppd'])

clfRandomForest.fit(X_train_sfsrf, y_trainrf)
y_pred_crf = clfRandomForest.predict(Xcdrn18)
y_pred_proba_crf = clfRandomForest.predict_proba(Xcdrn18)[:, 1]
confmat_test_crf = confusion_matrix(y_true=ycdrn18, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(ycdrn18, y_pred_crf))
print('the classification_report:', classification_report(ycdrn18, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(ycdrn18, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(ycdrn18, y_pred_proba_crf))
#withhistory
#confmat_test:
# [[42610  7849]
# [  978  2535]]
#the acc is: 0.8364522344919588
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.84      0.91     50459
#        1.0       0.24      0.72      0.36      3513
#
#avg / total       0.93      0.84      0.87     53972
#
#the auc of random forest is: 0.7944912077749656
#the brier socre is 0.14541212361391403
###########################################################################
#cdrn12
dfcdrn12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew12_2.csv', delimiter=',', header=0)
dfcdrn12 = pd.merge(dfcdrn12,visitexafter1y,on = ['person_id','condition_start_date'], how='inner')
df2cdrn12 = pd.DataFrame(dfcdrn12, columns=columnsrf2)

df2cdrn12[np.isnan(df2cdrn12)] = '0'
ycdrn12 = df2cdrn12['ppd']
Xcdrn12 = df2cdrn12.drop(columns=['ppd'])

clfRandomForest.fit(X_train_sfsrf, y_trainrf)
y_pred_crf = clfRandomForest.predict(Xcdrn12)
y_pred_proba_crf = clfRandomForest.predict_proba(Xcdrn12)[:, 1]
confmat_test_crf = confusion_matrix(y_true=ycdrn12, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(ycdrn12, y_pred_crf))
print('the classification_report:', classification_report(ycdrn12, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(ycdrn12, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(ycdrn12, y_pred_proba_crf))

#without history
#confmat_test:
# [[42699  7760]
# [ 1008  2505]]
#the acc is: 0.8375453939079522
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.85      0.91     50459
#        1.0       0.24      0.71      0.36      3513
#
#avg / total       0.93      0.84      0.87     53972
#
#the auc of random forest is: 0.7877055242608125
#the brier socre is 0.14424491340997303






