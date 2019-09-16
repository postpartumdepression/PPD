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
df = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodelwithhistory.csv', delimiter=',', header=0)
df['ppd'].value_counts()
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visitexafter1ywcmnew.csv', sep=',')
visitexafter1y = pd.DataFrame(visitexafter1y,columns=['person_id','condition_start_date'])
df_2 = pd.merge(df,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df_2['ppd'].value_counts()


#withouhistory
#clfRandomForest = ensemble.RandomForestClassifier(bootstrap=True, class_weight='balanced',
#                       criterion='gini', max_depth=8, max_features='auto',
#                       max_leaf_nodes=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=21,
#                       min_samples_split=140, min_weight_fraction_leaf=0.0,
#                       n_estimators=10, n_jobs=None, oob_score=False,
#                       random_state=None, verbose=0, warm_start=False)

clfRandomForest = ensemble.RandomForestClassifier(bootstrap=True, class_weight='balanced',random_state = 171,
                                                  criterion='gini', max_depth=8,min_impurity_decrease=0.0,
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
                                                  


columnsrf28 = ['ppd','dbp2nd', 'combine', 'C1209.0', 'C1253.0', 'C145.0', 'C1655.0', 'C1844.0', 'C209.0', 'C2168.0', 'C2471.0', 'C2517.0', 'C2644.0', 'C2906.0', 'C3174.0', 'C3267.0', 'C3447.0', 'C3624.0', 'C3626.0', 'C3734.0', 'C4126.0', 'C4179.0', 'C4468.0', 'C4764.0', 'C4774.0', 'M13.0', 'anxiety', 'mooddisorder', 'otherdisorder']



###########################################################################
columnsrf = columnsrf88 + columnsrf28
columnsrf = list(dict.fromkeys(columnsrf))

columnsrf_2 = pd.DataFrame(columnsrf, columns=['newvariable']) #change list to dataframe
##variable name
name_list = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/name_list.csv',delimiter=',', header=0)
columnsnamerf = pd.merge(columnsrf_2, name_list, on = ['newvariable'], how='left')
#columnsnamerf.to_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/result/selectedfeature1908/rffeaturelist.csv',header=0)
df3 = pd.DataFrame(df, columns=columnsrf)
df3_2 = df3
df3_2['a'] = 1
y3 = df3_2['ppd']
X3 = df3_2.drop(columns=['ppd'])
X_train_sfsrf3, X_test_sfsrf3, y_trainrf3, y_testrf3 = train_test_split(X3, y3, test_size=0.2, random_state=17)
dftrain = pd.concat([X_train_sfsrf3, y_trainrf3],axis=1)

for i in columnsrf:
    print(dftrain.groupby(['ppd',i])['a'].count())

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
# [[2725  108]
# [  16   70]]
#the acc is: 0.9575196985268928
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.96      0.98      2833
#        1.0       0.39      0.81      0.53        86
#
#avg / total       0.98      0.96      0.96      2919
#
#the auc of random forest is: 0.9093757952371961
#the brier socre is 0.08653701284499839

import pickle 
with open('/Users/bu/Desktop/clfrandomforest.pickle', 'wb') as f:
    pickle.dump(clfRandomForest, f)
    
###########################################################################
#feature name
columnsrf_2 = pd.DataFrame(columnsrf, columns=['newvariable']) #change list to dataframe
#variable name
name_list = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/name_list.csv',delimiter=',', header=0)

columnsnamerf = pd.merge(columnsrf_2, name_list, on = ['newvariable'], how='left')
columnsnamerf.to_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/result/selectedfeature1908/lgfeaturelist.csv',header=0)


###########################################################################

#wcmmodel30
dfwcmmodel30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel30.csv', delimiter=',', header=0)
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
# [[13747   431]
# [  117   299]]
#the acc is: 0.9624503220501576
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98     14178
#        1.0       0.41      0.72      0.52       416
#
#avg / total       0.97      0.96      0.97     14594
#
#the auc of random forest is: 0.879347879162733
#the brier socre is 0.07357716781084721
###########################################################################
#wcmmodel24
dfwcmmodel24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel24.csv', delimiter=',', header=0)
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
# [[2752   81]
# [  19   67]]
#the acc is: 0.9657416923603974
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2833
#        1.0       0.45      0.78      0.57        86
#
#avg / total       0.98      0.97      0.97      2919
#
#the auc of random forest is: 0.8675924937817581
#the brier socre is 0.07076265797152259

###########################################################################
#wcmmodel18
dfwcmmodel18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel18.csv', delimiter=',', header=0)
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

#withouthistory
#confmat_test:
# [[2752   81]
# [  19   67]]
#the acc is: 0.9657416923603974
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2833
#        1.0       0.45      0.78      0.57        86
#
#avg / total       0.98      0.97      0.97      2919
#
#the auc of random forest is: 0.8752431886651508
#the brier socre is 0.07046719057878313

###########################################################################
#wcmmodel12
dfwcmmodel12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel12.csv', delimiter=',', header=0)
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

##without history

#confmat_test:
# [[2753   80]
# [  19   67]]
#the acc is: 0.9660842754367934
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2833
#        1.0       0.46      0.78      0.58        86
#
#avg / total       0.98      0.97      0.97      2919
#
#the auc of random forest is: 0.8697514345052908
#the brier socre is 0.07037111447781703
###########################################################################
#cdrn
dfcdrn = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
dfcdrn = pd.merge(dfcdrn,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
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
# [[42613  7811]
# [  250  1639]]
#the acc is: 0.8459082828360064
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.85      0.91     50424
#        1.0       0.17      0.87      0.29      1889
#
#avg / total       0.96      0.85      0.89     52313
#
#the auc of random forest is: 0.876252691102164
#the brier socre is 0.16110032593504356

###########################################################################
#cdrn30
dfcdrn30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew30.csv', delimiter=',', header=0)
dfcdrn30 = pd.merge(dfcdrn30,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
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
#without history
#confmat_test:
# [[42743  7681]
# [  254  1635]]
#the acc is: 0.8483168619654771
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.85      0.92     50424
#        1.0       0.18      0.87      0.29      1889
#
#avg / total       0.96      0.85      0.89     52313
#
#the auc of random forest is: 0.8751662608333843
#the brier socre is 0.1565156815885284
###########################################################################
#cdrn24
dfcdrn24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew24.csv', delimiter=',', header=0)
dfcdrn24 = pd.merge(dfcdrn24,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
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
# [[42800  7624]
# [  255  1634]]
#the acc is: 0.8493873415785751
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.85      0.92     50424
#        1.0       0.18      0.87      0.29      1889
#
#avg / total       0.96      0.85      0.89     52313
#
#the auc of random forest is: 0.8763639236049082
#the brier socre is 0.1556857223433604

###########################################################################
#cdrn18
dfcdrn18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew18.csv', delimiter=',', header=0)
dfcdrn18 = pd.merge(dfcdrn18,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
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
# [[42839  7585]
# [  256  1633]]
#the acc is: 0.8501137384588917
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.85      0.92     50424
#        1.0       0.18      0.86      0.29      1889
#
#avg / total       0.96      0.85      0.89     52313
#
#the auc of random forest is: 0.8760457849989002
#the brier socre is 0.15517454296458638
############################################################################
#cdrn12
dfcdrn12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew12.csv', delimiter=',', header=0)
dfcdrn12 = pd.merge(dfcdrn12,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
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
# [[42871  7553]
# [  258  1631]]
#the acc is: 0.8506872096801942
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.85      0.92     50424
#        1.0       0.18      0.86      0.29      1889
#
#avg / total       0.96      0.85      0.89     52313
#
#the auc of random forest is: 0.8758205221206434
#the brier socre is 0.15482587165056932










