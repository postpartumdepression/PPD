#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:43:22 2019

@author: bu
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, accuracy_score, auc, precision_recall_fscore_support
from sklearn import ensemble
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import brier_score_loss

###########################################################################
df = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodelwithhistory_2.csv', delimiter=',', header=0)
df['ppd'].value_counts()
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visitexafter1ywcmnew_2.csv', sep=',')
visitexafter1y = pd.DataFrame(visitexafter1y,columns=['person_id','condition_start_date'])
df_2 = pd.merge(df,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df_2['ppd'].value_counts()

df_2.isnull().any()

# 0.0    14181
# 1.0      416

#without history
clfxgboost = XGBClassifier(base_score=0.5, booster='gbtree', class_weight='balanced',colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                           gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=5,
                           min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,
                           objective='binary:logistic', random_state=0,
                           reg_alpha=0, reg_lambda=1, scale_pos_weight=39.58894230769231,
                           subsample=1, verbosity=1)

#with history
#clfXGboost = XGBClassifier(base_score=0.5, booster='gbtree', class_weight='balanced',
#              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
#              gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=5,
#              min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,
#              nthread=None, objective='binary:logistic', random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=39.58894230769231,
#              seed=None, silent=None, subsample=1, verbosity=1)




columnsxgboost12 = ['ppd','dbp2nd', 'C1465.0', 'C1949.0', 'C2057.0', 'C2220.0', 'C2517.0', 'C406.0', 'M42.0', 'M56.0', 'anxiety', 'mooddisorder', 'otherdisorder']
columnsxgboost17 = ['ppd','dbp2nd', 'asian', 'C1203.0', 'C1465.0', 'C1537.0', 'C1685.0', 'C1764.0', 'C2070.0', 'C3685.0', 'C3755.0', 'C4056.0', 'C4101.0', 'C419.0', 'M133.0', 'M14.0', 'M32.0', 'M98.0']


###########################################################################
columnsxgboost = columnsxgboost12 + columnsxgboost17
columnsxgboost = list(dict.fromkeys(columnsxgboost))

#columnsxgboost_2 = pd.DataFrame(columnsxgboost, columns=['newvariable']) #change list to dataframe
#variable name
#name_list = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/name_list.csv',delimiter=',', header=0)
#columnsnamexgboost = pd.merge(columnsxgboost_2, name_list, on = ['newvariable'], how='left')

columnsxgboost2 = ['ppd',
 'dbp2nd',
 #'C1465.0',Disorder of ear
 #'C1949.0',Finding related to blood, organ, or tissue donation
 #'C2057.0',#Genetic finding
 #'C2220.0',Hemorrhoids in pregnancy
 #'C2517.0',Inflammation of cervix
 #'C406.0',Antenatal ultrasound scan abnormal
 #'M42.0',BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS
 'M56.0',#DRUGS FOR FUNCTIONAL GASTROINTESTINAL DISORDERS
 'anxiety',
 'mooddisorder',
 'otherdisorder',
 'asian',
 #'C1203.0',Contact dermatitis
 'C1537.0',#Disorder of pregnancy
 'C1685.0',#Elderly primigravida
 #'C1764.0',Ethnic background
 #'C2070.0',Genitourinary tract hemorrhage
 'C3685.0',#Post-term pregnancy
 #'C3755.0',Pregnancy with uncertain dates
 # 'C4056.0',#Rheumatoid arthritis
 # 'C4101.0',#Scar conditions and fibrosis of skin
 'C419.0',#Anxiety
 'M133.0',#TOPICAL PRODUCTS FOR JOINT AND MUSCULAR PAIN
 'M14.0',#ANTIDEPRESSANTS
 #'M32.0',ANTIPSYCHOTICS
 'M98.0']#OTHER PLAIN VITAMIN PREPARATIONS

#wcm
df3 = pd.DataFrame(df_2, columns=columnsxgboost2)
df3.isnull().any()
#df3[np.isnan(df3)] = '0'
#df3.head(2)
y = df3['ppd']
X = df3.drop(columns=['ppd'])
X_train_sfsxgboost, X_test_sfsxgboost, y_trainxgboost, y_testxgboost = train_test_split(X, y, test_size=0.2, random_state=17)
print(X_test_sfsxgboost.shape)
print(X_train_sfsxgboost.shape)

# from xgboost import plot_tree
# import matplotlib.pyplot as plt
# plot_tree(clfxgboost,num_trees=10)#, rankdir='LR'
# plt.show()

clfxgboost.fit(X_train_sfsxgboost, y_trainxgboost)
y_pred_c = clfxgboost.predict(X_test_sfsxgboost)
y_pred_proba_cxg = clfxgboost.predict_proba(X_test_sfsxgboost)[:, 1]
confmat_test_c = confusion_matrix(y_true=y_testxgboost, y_pred=y_pred_c)

print('confmat_test:\n', confmat_test_c)
print('the acc is:', accuracy_score(y_testxgboost, y_pred_c))
print('the classification_report:\n', classification_report(y_testxgboost, y_pred_c))
print('the auc of xgboost is:', roc_auc_score(y_testxgboost, y_pred_proba_cxg))
print('the brier score is',brier_score_loss(y_testxgboost, y_pred_proba_cxg))

#fit withouthistory
# confmat_test:
#  [[2684  179]
#  [  23  154]]
# the acc is: 0.9335526315789474
# the classification_report:
#                precision    recall  f1-score   support
#          0.0       0.99      0.94      0.96      2863
#          1.0       0.46      0.87      0.60       177
#    micro avg       0.93      0.93      0.93      3040
#    macro avg       0.73      0.90      0.78      3040
# weighted avg       0.96      0.93      0.94      3040
# the auc of xgboost is: 0.9351575033892385
# the brier socre is 0.10085559301347664

import pickle
with open('/Users/bu/Desktop/clfxgboost_2.pickle', 'wb') as f:
    pickle.dump(clfxgboost, f)
with open('/Users/bu/Desktop/clfxgboost_2.pickle', 'rb') as f:
    clf2 = pickle.load(f)
     #测试读取后的Model
     print(clf2.predict(X[0:1])

###########################################################################
#wcmmodel30
dfwcmmodel30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel30_2.csv', delimiter=',', header=0)
dfwcmmodel30 = pd.merge(dfwcmmodel30,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel30 = pd.DataFrame(dfwcmmodel30, columns=columnsxgboost2)
df2wcmmodel30[np.isnan(df2wcmmodel30)] = 0
y30 = df2wcmmodel30['ppd']
X30 = df2wcmmodel30.drop(columns=['ppd'])
X_train30, X_test30, y_train30, y_test30 = train_test_split(X30, y30, test_size=0.2, random_state=17)

clfxgboost.fit(X_train_sfsxgboost, y_trainxgboost)
y_pred_cxgboost = clfxgboost.predict(X_test30)
y_pred_proba_cxgboost = clfxgboost.predict_proba(X_test30)[:, 1]
confmat_test_cxgboost = confusion_matrix(y_true=y_test30, y_pred=y_pred_cxgboost)

print('confmat_test:\n', confmat_test_cxgboost)
print('the acc is:', accuracy_score(y_test30, y_pred_cxgboost))
print('the classification_report:', classification_report(y_test30, y_pred_cxgboost))
print('the auc of xgboost is:', roc_auc_score(y_test30, y_pred_proba_cxgboost))
print('the brier socre is',brier_score_loss(y_test30, y_pred_proba_cxgboost))
# confmat_test:
#  [[2751  112]
#  [  31  146]]
# the acc is: 0.9529605263157894
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.99      0.96      0.97      2863
#          1.0       0.57      0.82      0.67       177
#    micro avg       0.95      0.95      0.95      3040
#    macro avg       0.78      0.89      0.82      3040
# weighted avg       0.96      0.95      0.96      3040
# the auc of xgboost is: 0.9120583876499504
# the brier socre is 0.08499774967949507


###########################################################################
#wcmmodel24
dfwcmmodel24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel24_2.csv', delimiter=',', header=0)
dfwcmmodel24 = pd.merge(dfwcmmodel24,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel24 = pd.DataFrame(dfwcmmodel24, columns=columnsxgboost2)
df2wcmmodel24[np.isnan(df2wcmmodel24)] = 0
y24 = df2wcmmodel24['ppd']
X24 = df2wcmmodel24.drop(columns=['ppd'])
X_train24, X_test24, y_train24, y_test24 = train_test_split(X24, y24, test_size=0.2, random_state=17)

clfxgboost.fit(X_train_sfsxgboost, y_trainxgboost)
y_pred_cxgboost = clfxgboost.predict(X_test24)
y_pred_proba_cxgboost = clfxgboost.predict_proba(X_test24)[:, 1]
confmat_test_cxgboost = confusion_matrix(y_true=y_test24, y_pred=y_pred_cxgboost)

print('confmat_test:\n', confmat_test_cxgboost)
print('the acc is:', accuracy_score(y_test24, y_pred_cxgboost))
print('the classification_report:', classification_report(y_test24, y_pred_cxgboost))
print('the auc of mlp is:', roc_auc_score(y_test24, y_pred_proba_cxgboost))
print('the brier socre is',brier_score_loss(y_test24, y_pred_proba_cxgboost))

#withouthistory
# confmat_test:
#  [[2752  111]
#  [  30  147]]
# the acc is: 0.9536184210526316
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.99      0.96      0.98      2863
#          1.0       0.57      0.83      0.68       177
#    micro avg       0.95      0.95      0.95      3040
#    macro avg       0.78      0.90      0.83      3040
# weighted avg       0.96      0.95      0.96      3040
# the auc of mlp is: 0.9185862484731161
# the brier socre is 0.08198482388200991

###########################################################################
#wcmmodel18
dfwcmmodel18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel18_2.csv', delimiter=',', header=0)
dfwcmmodel18 = pd.merge(dfwcmmodel18,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel18 = pd.DataFrame(dfwcmmodel18, columns=columnsxgboost2)
df2wcmmodel18[np.isnan(df2wcmmodel18)] = 0
y18 = df2wcmmodel18['ppd']
X18 = df2wcmmodel18.drop(columns=['ppd'])
X_train18, X_test18, y_train18, y_test18 = train_test_split(X18, y18, test_size=0.2, random_state=17)

clfxgboost.fit(X_train_sfsxgboost, y_trainxgboost)
y_pred_cxgboost = clfxgboost.predict(X_test18)
y_pred_proba_cxgboost = clfxgboost.predict_proba(X_test18)[:, 1]
confmat_test_cxgboost = confusion_matrix(y_true=y_test18, y_pred=y_pred_cxgboost)

print('confmat_test:\n', confmat_test_cxgboost)
print('the acc is:', accuracy_score(y_test18, y_pred_cxgboost))
print('the classification_report:', classification_report(y_test18, y_pred_cxgboost))
print('the auc of xbgoost is:', roc_auc_score(y_test18, y_pred_proba_cxgboost))
print('the brier socre is',brier_score_loss(y_test18, y_pred_proba_cxgboost))

# confmat_test:
#  [[2764   99]
#  [  31  146]]
# the acc is: 0.9572368421052632
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.99      0.97      0.98      2863
#          1.0       0.60      0.82      0.69       177
#    micro avg       0.96      0.96      0.96      3040
#    macro avg       0.79      0.90      0.83      3040
# weighted avg       0.97      0.96      0.96      3040
# the auc of xbgoost is: 0.9020209136242455
# the brier socre is 0.0974604128208319

###########################################################################
#wcmmodel12
dfwcmmodel12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel12_2.csv', delimiter=',', header=0)
dfwcmmodel12 = pd.merge(dfwcmmodel12,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel12 = pd.DataFrame(dfwcmmodel12, columns=columnsxgboost2)
df2wcmmodel12[np.isnan(df2wcmmodel12)] = 0
y12 = df2wcmmodel12['ppd']
X12 = df2wcmmodel12.drop(columns=['ppd'])
X_train12, X_test12, y_train12, y_test12 = train_test_split(X12, y12, test_size=0.2, random_state=17)

clfxgboost.fit(X_train_sfsxgboost, y_trainxgboost)
y_pred_cxgboost = clfxgboost.predict(X_test12)
y_pred_proba_cxgboost = clfxgboost.predict_proba(X_test12)[:, 1]
confmat_test_cxgboost = confusion_matrix(y_true=y_test12, y_pred=y_pred_cxgboost)

print('confmat_test:\n', confmat_test_cxgboost)
print('the acc is:', accuracy_score(y_test12, y_pred_cxgboost))
print('the classification_report:', classification_report(y_test12, y_pred_cxgboost))
print('the auc of xbgoost is:', roc_auc_score(y_test12, y_pred_proba_cxgboost))
print('the brier socre is',brier_score_loss(y_test12, y_pred_proba_cxgboost))

# confmat_test:
#  [[2767   96]
#  [  31  146]]
# the acc is: 0.9582236842105263
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.99      0.97      0.98      2863
#          1.0       0.60      0.82      0.70       177
#    micro avg       0.96      0.96      0.96      3040
#    macro avg       0.80      0.90      0.84      3040
# weighted avg       0.97      0.96      0.96      3040
# the auc of xbgoost is: 0.9076578043259905
# the brier socre is 0.0680924979378865

###########################################################################
#cdrn
dfcdrn = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew_2.csv', delimiter=',', header=0)
dfcdrn['ppd'].value_counts()
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew_2.csv', delimiter=',', header=0)

dfcdrn = pd.merge(dfcdrn,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn = pd.DataFrame(dfcdrn, columns=columnsxgboost2)
df2cdrn[np.isnan(df2cdrn)] = 0
ycdrn = df2cdrn['ppd']
Xcdrn = df2cdrn.drop(columns=['ppd'])

clfxgboost.fit(X_train_sfsxgboost, y_trainxgboost)
y_pred_ccdrn = clfxgboost.predict(Xcdrn)
y_pred_proba_cdtcdrn = clfxgboost.predict_proba(Xcdrn)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn, y_pred_ccdrn))
print('the classification_report:\n', classification_report(ycdrn, y_pred_ccdrn))
print('the auc of logistics is:', roc_auc_score(ycdrn, y_pred_proba_cdtcdrn))
print('the brier socre is',brier_score_loss(ycdrn, y_pred_proba_cdtcdrn))

#fit withouthistory
# confmat_test:
#  [[42470  7989]
#  [  545  2968]]
# the acc is: 0.8418809753205366
# the classification_report:
#                precision    recall  f1-score   support
#          0.0       0.99      0.84      0.91     50459
#          1.0       0.27      0.84      0.41      3513
#    micro avg       0.84      0.84      0.84     53972
#    macro avg       0.63      0.84      0.66     53972
# weighted avg       0.94      0.84      0.88     53972
# the auc of logistics is: 0.8636117565711191
# the brier socre is 0.17759427234707595
###########################################################################
#cdrn30
dfcdrn30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew30_2.csv', delimiter=',', header=0)
dfcdrn30 = pd.merge(dfcdrn30,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn30 = pd.DataFrame(dfcdrn30, columns=columnsxgboost2)

df2cdrn30[np.isnan(df2cdrn30)] = 0
ycdrn30 = df2cdrn30['ppd']
Xcdrn30 = df2cdrn30.drop(columns=['ppd'])

clfxgboost.fit(X_train_sfsxgboost, y_trainxgboost)
y_pred_ccdrn = clfxgboost.predict(Xcdrn30)
y_pred_proba_cdtcdrn = clfxgboost.predict_proba(Xcdrn30)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn30, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn30, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn30, y_pred_ccdrn))
print('the auc of logistics is:', roc_auc_score(ycdrn30, y_pred_proba_cdtcdrn))
print('the brier socre is',brier_score_loss(ycdrn30, y_pred_proba_cdtcdrn))

# confmat_test:
#  [[42581  7878]
#  [  934  2579]]
# the acc is: 0.8367301563773809
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.98      0.84      0.91     50459
#          1.0       0.25      0.73      0.37      3513
#    micro avg       0.84      0.84      0.84     53972
#    macro avg       0.61      0.79      0.64     53972
# weighted avg       0.93      0.84      0.87     53972
# the auc of logistics is: 0.8014954570162899
# the brier socre is 0.18102932169925262
###########################################################################
#cdrn24
dfcdrn24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew24_2.csv', delimiter=',', header=0)
dfcdrn24 = pd.merge(dfcdrn24,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn24 = pd.DataFrame(dfcdrn24, columns=columnsxgboost2)

df2cdrn24[np.isnan(df2cdrn24)] = 0
ycdrn24 = df2cdrn24['ppd']
Xcdrn24 = df2cdrn24.drop(columns=['ppd'])

clfxgboost.fit(X_train_sfsxgboost, y_trainxgboost)
y_pred_ccdrn = clfxgboost.predict(Xcdrn24)
y_pred_proba_cdtcdrn = clfxgboost.predict_proba(Xcdrn24)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn24, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn24, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn24, y_pred_ccdrn))
print('the auc of logistics is:', roc_auc_score(ycdrn24, y_pred_proba_cdtcdrn))
print('the brier socre is',brier_score_loss(ycdrn24, y_pred_proba_cdtcdrn))

# confmat_test:
#  [[42655  7804]
#  [  953  2560]]
# the acc is: 0.8377492032905951
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.98      0.85      0.91     50459
#          1.0       0.25      0.73      0.37      3513
#    micro avg       0.84      0.84      0.84     53972
#    macro avg       0.61      0.79      0.64     53972
# weighted avg       0.93      0.84      0.87     53972
# the auc of logistics is: 0.797800303659319
# the brier socre is 0.18025895671627476

###########################################################################
#cdrn18
dfcdrn18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew18_2.csv', delimiter=',', header=0)
dfcdrn18 = pd.merge(dfcdrn18,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn18 = pd.DataFrame(dfcdrn18, columns=columnsxgboost2)

df2cdrn18[np.isnan(df2cdrn18)] = 0
ycdrn18 = df2cdrn18['ppd']
Xcdrn18 = df2cdrn18.drop(columns=['ppd'])

clfxgboost.fit(X_train_sfsxgboost, y_trainxgboost)
y_pred_ccdrn = clfxgboost.predict(Xcdrn18)
y_pred_proba_cdtcdrn = clfxgboost.predict_proba(Xcdrn18)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn18, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn18, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn18, y_pred_ccdrn))
print('the auc of logistics is:', roc_auc_score(ycdrn18, y_pred_proba_cdtcdrn))
print('the brier socre is',brier_score_loss(ycdrn18, y_pred_proba_cdtcdrn))

# confmat_test:
#  [[42711  7748]
#  [  990  2523]]
# the acc is: 0.8381012376787964
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.98      0.85      0.91     50459
#          1.0       0.25      0.72      0.37      3513
#    micro avg       0.84      0.84      0.84     53972
#    macro avg       0.61      0.78      0.64     53972
# weighted avg       0.93      0.84      0.87     53972
# the auc of logistics is: 0.7931355852110531
# the brier socre is 0.1799037250275233

###########################################################################
#cdrn12
dfcdrn12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew12_2.csv', delimiter=',', header=0)
dfcdrn12 = pd.merge(dfcdrn12,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn12 = pd.DataFrame(dfcdrn12, columns=columnsxgboost2)

df2cdrn12[np.isnan(df2cdrn12)] = 0
ycdrn12 = df2cdrn12['ppd']
Xcdrn12 = df2cdrn12.drop(columns=['ppd'])

clfxgboost.fit(X_train_sfsxgboost, y_trainxgboost)
y_pred_ccdrn = clfxgboost.predict(Xcdrn12)
y_pred_proba_cdtcdrn = clfxgboost.predict_proba(Xcdrn12)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn12, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn12, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn12, y_pred_ccdrn))
print('the auc of logistics is:', roc_auc_score(ycdrn12, y_pred_proba_cdtcdrn))
print('the brier socre is',brier_score_loss(ycdrn12, y_pred_proba_cdtcdrn))

# confmat_test:
#  [[42760  7699]
#  [ 1019  2494]]
# the acc is: 0.8384718001926925
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.98      0.85      0.91     50459
#          1.0       0.24      0.71      0.36      3513
#    micro avg       0.84      0.84      0.84     53972
#    macro avg       0.61      0.78      0.64     53972
# weighted avg       0.93      0.84      0.87     53972
# the auc of logistics is: 0.7889008647273312
# the brier socre is 0.17958266006620635

