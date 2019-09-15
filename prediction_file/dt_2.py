#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:15:52 2019

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

#withouthistroy
clfDecisionTree = tree.DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=8,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=230,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')


columnsdt18 = ['ppd','dbp2nd', 'C1425.0', 'C1537.0', 'C1834.0', 'C1957.0', 'C2213.0', 'C3675.0', 'C3729.0', 'C4110.0', 'C419.0', 'C539.0', 'C689.0', 'C904.0', 'M131.0', 'M14.0', 'M21.0', 'M33.0', 'M55.0']


#withhistroy
#clfDecisionTree = tree.DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=13,
#                       max_features=None, max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=290,
#                       min_weight_fraction_leaf=0.0, presort=False,
#                       random_state=None, splitter='best')

columnsdt18_2 = ['ppd','weight1st', 'dbp2nd', 'C1078.0', 'C1387.0', 'C2517.0', 'C2644.0', 'C3243.0', 'C3432.0', 'C3795.0', 'C406.0', 'C4187.0', 'C4429.0', 'C4728.0', 'C4743.0', 'M25.0', 'anxiety', 'mooddisorder', 'otherdisorder']

###########################################################################
columnsdt = columnsdt18 + columnsdt18_2
columnsdt = list(dict.fromkeys(columnsdt))

#wcm
columnsdt2 = ['ppd',
 'dbp2nd',
 'C1425.0',#Disease of the respiratory system complicating pregnancy, childbirth and/or the puerperium
 'C1537.0',#Disorder of pregnancy
 'C1834.0',#Female infertility
 'C1957.0',#First degree perineal tear during delivery - delivered
 'C2213.0',#Hemorrhage in early pregnancy
 'C3675.0',#Positions of breech presentation - finding
 #'C3729.0',
 #'C4110.0',
 'C419.0',#Anxiety
 'C539.0',#Baby overdue
 'C689.0',#Blood coagulation disorder
 'C904.0',#Chorioamnionitis
 'M131.0',#POSTERIOR PITUITARY LOBE HORMONES
 'M14.0',#ANTIDEPRESSANTS
 'M21.0',#ANTIHISTAMINES FOR SYSTEMIC USE
 #'M33.0',
 'M55.0',#DRUGS FOR CONSTIPATION
 'weight1st',
 'C1078.0',#Complication occurring during pregnancy
 'C1387.0',#Diarrhea
 #'C2517.0',
 #'C2644.0',
 'C3243.0',#Non-proteinuric hypertension of pregnancy
 #'C3432.0',
 #'C3795.0',
 #'C406.0',
 #'C4187.0',
 #'C4429.0',
 'C4728.0',#Urinary tract infectious disease
 'C4743.0',#Uterine fibroids in pregnancy, childbirth and the puerperium
 'M25.0',#ANTIINFLAMMATORY AND ANTIRHEUMATIC PRODUCTS, NON-STEROIDS
 'anxiety',
 'mooddisorder',
 'otherdisorder']

df3 = pd.DataFrame(df_2, columns=columnsdt2)
df3.isnull().any()
#df3[np.isnan(df3)] = '0'
#df3.head(2)
y = df3['ppd']
X = df3.drop(columns=['ppd'])
X_train_sfsdt, X_test_sfsdt, y_traindt, y_testdt = train_test_split(X, y, test_size=0.2, random_state=17)
print(X_test_sfsdt.shape)
print(X_train_sfsdt.shape)

clfDecisionTree.fit(X_train_sfsdt, y_traindt)
y_pred_cdt = clfDecisionTree.predict(X_test_sfsdt)
y_pred_proba_cdt = clfDecisionTree.predict_proba(X_test_sfsdt)[:, 1]
confmat_test_cdt = confusion_matrix(y_true=y_testdt, y_pred=y_pred_cdt)

print('confmat_test:\n', confmat_test_cdt)
print('the acc is:', accuracy_score(y_testdt, y_pred_cdt))
print('the classification_report:', classification_report(y_testdt, y_pred_cdt))
print('the auc of random forest is:', roc_auc_score(y_testdt, y_pred_proba_cdt))
print('the brier socre is',brier_score_loss(y_testdt, y_pred_proba_cdt))

#fit withouthistory
#confmat_test:
# [[2736  127]
# [  23  154]]
#the acc is: 0.9506578947368421
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.96      0.97      2863
#        1.0       0.55      0.87      0.67       177
#
#avg / total       0.97      0.95      0.96      3040
#
#the auc of random forest is: 0.9110125091021033
#the brier socre is 0.051837218561148514
import pickle 
with open('/Users/bu/Desktop/clfdecisiontree_2.pickle', 'wb') as f:
    pickle.dump(clfDecisionTree, f)
###########################################################################
#wcmmodel30
dfwcmmodel30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel30_2.csv', delimiter=',', header=0)
dfwcmmodel30 = pd.merge(dfwcmmodel30,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel30 = pd.DataFrame(dfwcmmodel30, columns=columnsdt2)
df2wcmmodel30[np.isnan(df2wcmmodel30)] = '0'
y30 = df2wcmmodel30['ppd']
X30 = df2wcmmodel30.drop(columns=['ppd'])
X_train30, X_test30, y_train30, y_test30 = train_test_split(X30, y30, test_size=0.2, random_state=17)

clfDecisionTree.fit(X_train_sfsdt, y_traindt)
y_pred_cdt = clfDecisionTree.predict(X_test30)
y_pred_proba_cdt = clfDecisionTree.predict_proba(X_test30)[:, 1]
confmat_test_cdt = confusion_matrix(y_true=y_test30, y_pred=y_pred_cdt)

print('confmat_test:\n', confmat_test_cdt)
print('the acc is:', accuracy_score(y_test30, y_pred_cdt))
print('the classification_report:', classification_report(y_test30, y_pred_cdt))
print('the auc of random forest is:', roc_auc_score(y_test30, y_pred_proba_cdt))
print('the brier socre is',brier_score_loss(y_test30, y_pred_proba_cdt))

#fit without history

#confmat_test:
# [[2760  103]
# [  31  146]]
#the acc is: 0.9559210526315789
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.96      0.98      2863
#        1.0       0.59      0.82      0.69       177
#
#avg / total       0.97      0.96      0.96      3040
#
#the auc of random forest is: 0.8870401834431505
#the brier socre is 0.04847307294762688
###########################################################################
#wcmmodel24
dfwcmmodel24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel24_2.csv', delimiter=',', header=0)
dfwcmmodel24 = pd.merge(dfwcmmodel24,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel24 = pd.DataFrame(dfwcmmodel24, columns=columnsdt2)
df2wcmmodel24[np.isnan(df2wcmmodel24)] = '0'
y24 = df2wcmmodel24['ppd']
X24 = df2wcmmodel24.drop(columns=['ppd'])
X_train24, X_test24, y_train24, y_test24 = train_test_split(X24, y24, test_size=0.2, random_state=17)

clfDecisionTree.fit(X_train_sfsdt, y_traindt)
y_pred_cdt = clfDecisionTree.predict(X_test24)
y_pred_proba_cdt = clfDecisionTree.predict_proba(X_test24)[:, 1]
confmat_test_cdt = confusion_matrix(y_true=y_test24, y_pred=y_pred_cdt)

print('confmat_test:\n', confmat_test_cdt)
print('the acc is:', accuracy_score(y_test24, y_pred_cdt))
print('the classification_report:', classification_report(y_test24, y_pred_cdt))
print('the auc of random forest is:', roc_auc_score(y_test24, y_pred_proba_cdt))
print('the brier socre is',brier_score_loss(y_test24, y_pred_proba_cdt))

#withouthistroy

#confmat_test:
# [[2760  103]
# [  30  147]]
#the acc is: 0.95625
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.96      0.98      2863
#        1.0       0.59      0.83      0.69       177
#
#avg / total       0.97      0.96      0.96      3040
#
#the auc of random forest is: 0.8953065706826429
#the brier socre is 0.047601340851413716

###########################################################################
#wcmmodel18
dfwcmmodel18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel18_2.csv', delimiter=',', header=0)
dfwcmmodel18 = pd.merge(dfwcmmodel18,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel18 = pd.DataFrame(dfwcmmodel18, columns=columnsdt2)
df2wcmmodel18[np.isnan(df2wcmmodel18)] = '0'
y18 = df2wcmmodel18['ppd']
X18 = df2wcmmodel18.drop(columns=['ppd'])
X_train18, X_test18, y_train18, y_test18 = train_test_split(X18, y18, test_size=0.2, random_state=17)

clfDecisionTree.fit(X_train_sfsdt, y_traindt)
y_pred_cdt = clfDecisionTree.predict(X_test18)
y_pred_proba_cdt = clfDecisionTree.predict_proba(X_test18)[:, 1]
confmat_test_cdt = confusion_matrix(y_true=y_test18, y_pred=y_pred_cdt)

print('confmat_test:\n', confmat_test_cdt)
print('the acc is:', accuracy_score(y_test18, y_pred_cdt))
print('the classification_report:', classification_report(y_test18, y_pred_cdt))
print('the auc of random forest is:', roc_auc_score(y_test18, y_pred_proba_cdt))
print('the brier socre is',brier_score_loss(y_test18, y_pred_proba_cdt))

#without history

#confmat_test:
# [[2762  101]
# [  31  146]]
#the acc is: 0.9565789473684211
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.96      0.98      2863
#        1.0       0.59      0.82      0.69       177
#
#avg / total       0.97      0.96      0.96      3040
#
#the auc of random forest is: 0.890260700028219
#the brier socre is 0.04775458521653261
###########################################################################
#wcmmodel12
dfwcmmodel12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel12_2.csv', delimiter=',', header=0)
dfwcmmodel12 = pd.merge(dfwcmmodel12,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel12 = pd.DataFrame(dfwcmmodel12, columns=columnsdt2)
df2wcmmodel12[np.isnan(df2wcmmodel12)] = '0'
y12 = df2wcmmodel12['ppd']
X12 = df2wcmmodel12.drop(columns=['ppd'])
X_train12, X_test12, y_train12, y_test12 = train_test_split(X12, y12, test_size=0.2, random_state=17)

clfDecisionTree.fit(X_train_sfsdt, y_traindt)
y_pred_cdt = clfDecisionTree.predict(X_test12)
y_pred_proba_cdt = clfDecisionTree.predict_proba(X_test12)[:, 1]
confmat_test_cdt = confusion_matrix(y_true=y_test12, y_pred=y_pred_cdt)

print('confmat_test:\n', confmat_test_cdt)
print('the acc is:', accuracy_score(y_test12, y_pred_cdt))
print('the classification_report:', classification_report(y_test12, y_pred_cdt))
print('the auc of random forest is:', roc_auc_score(y_test12, y_pred_proba_cdt))
print('the brier socre is',brier_score_loss(y_test12, y_pred_proba_cdt))

#without history

#confmat_test:
# [[2762  101]
# [  30  147]]
#the acc is: 0.9569078947368421
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.96      0.98      2863
#        1.0       0.59      0.83      0.69       177
#
#avg / total       0.97      0.96      0.96      3040
#
#the auc of random forest is: 0.9031082326428562
#the brier socre is 0.045168591832421015
###########################################################################
#cdrn
dfcdrn = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew_2.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew_2.csv', delimiter=',', header=0)
dfcdrn = pd.merge(dfcdrn,visitexafter1y,on = ['person_id','condition_start_date'], how='inner')

df2cdrn = pd.DataFrame(dfcdrn, columns=columnsdt2)
df2cdrn[np.isnan(df2cdrn)] = '0'
ycdrn = df2cdrn['ppd']
Xcdrn = df2cdrn.drop(columns=['ppd'])

clfDecisionTree.fit(X_train_sfsdt, y_traindt)
y_pred_ccdrn = clfDecisionTree.predict(Xcdrn)
y_pred_proba_cdtcdrn = clfDecisionTree.predict_proba(Xcdrn)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn, y_pred_ccdrn))
print('the auc of logistics is:', roc_auc_score(ycdrn, y_pred_proba_cdtcdrn))
print('the brier socre is',brier_score_loss(ycdrn, y_pred_proba_cdtcdrn))

#fit without history

#confmat_test:
# [[42357  8102]
# [  498  3015]]
#the acc is: 0.8406581190246795
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.84      0.91     50459
#        1.0       0.27      0.86      0.41      3513
#
#avg / total       0.94      0.84      0.88     53972
#
#the auc of logistics is: 0.8561584641603797
#the brier socre is 0.14927819963602307
###########################################################################
#cdrn30
dfcdrn30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew30_2.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew_2.csv', delimiter=',', header=0)
dfcdrn30 = pd.merge(dfcdrn30,visitexafter1y,on = ['person_id','condition_start_date'], how='inner')

df2cdrn30 = pd.DataFrame(dfcdrn30, columns=columnsdt2)

df2cdrn30[np.isnan(df2cdrn30)] = '0'
ycdrn30 = df2cdrn30['ppd']
Xcdrn30 = df2cdrn30.drop(columns=['ppd'])

clfDecisionTree.fit(X_train_sfsdt, y_traindt)
y_pred_ccdrn = clfDecisionTree.predict(Xcdrn30)
y_pred_proba_cdtcdrn = clfDecisionTree.predict_proba(Xcdrn30)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn30, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn30, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn30, y_pred_ccdrn))
print('the auc of logistics is:', roc_auc_score(ycdrn30, y_pred_proba_cdtcdrn))
print('the brier socre is',brier_score_loss(ycdrn30, y_pred_proba_cdtcdrn))

##fit without history
#confmat_test:
# [[42500  7959]
# [  937  2576]]
#the acc is: 0.8351737938190172
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.84      0.91     50459
#        1.0       0.24      0.73      0.37      3513
#
#avg / total       0.93      0.84      0.87     53972
#
#the auc of logistics is: 0.8017463476913022
#the brier socre is 0.1523764538567156

###########################################################################
#cdrn24
dfcdrn24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew24_2.csv', delimiter=',', header=0)
dfcdrn24 = pd.merge(dfcdrn24,visitexafter1y,on = ['person_id','condition_start_date'], how='inner')

df2cdrn24 = pd.DataFrame(dfcdrn24, columns=columnsdt2)

df2cdrn24[np.isnan(df2cdrn24)] = '0'
ycdrn24 = df2cdrn24['ppd']
Xcdrn24 = df2cdrn24.drop(columns=['ppd'])

clfDecisionTree.fit(X_train_sfsdt, y_traindt)
y_pred_ccdrn = clfDecisionTree.predict(Xcdrn24)
y_pred_proba_cdtcdrn = clfDecisionTree.predict_proba(Xcdrn24)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn24, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn24, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn24, y_pred_ccdrn))
print('the auc of dt is:', roc_auc_score(ycdrn24, y_pred_proba_cdtcdrn))
print('the brier socre is',brier_score_loss(ycdrn24, y_pred_proba_cdtcdrn))

#withouthistory

#confmat_test:
# [[42576  7883]
# [  957  2556]]
#the acc is: 0.8362113688579264
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.84      0.91     50459
#        1.0       0.24      0.73      0.37      3513
#
#avg / total       0.93      0.84      0.87     53972
#
#the auc of dt is: 0.798821960996402
#the brier socre is 0.1518028046065607
###########################################################################
#cdrn18
dfcdrn18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew18_2.csv', delimiter=',', header=0)
dfcdrn18 = pd.merge(dfcdrn18,visitexafter1y,on = ['person_id','condition_start_date'], how='inner')
df2cdrn18 = pd.DataFrame(dfcdrn18, columns=columnsdt2)

df2cdrn18[np.isnan(df2cdrn18)] = '0'
ycdrn18 = df2cdrn18['ppd']
Xcdrn18 = df2cdrn18.drop(columns=['ppd'])

clfDecisionTree.fit(X_train_sfsdt, y_traindt)
y_pred_ccdrn = clfDecisionTree.predict(Xcdrn18)
y_pred_proba_cdtcdrn = clfDecisionTree.predict_proba(Xcdrn18)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn18, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn18, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn18, y_pred_ccdrn))
print('the auc of dt is:', roc_auc_score(ycdrn18, y_pred_proba_cdtcdrn))
print('the brier socre is',brier_score_loss(ycdrn18, y_pred_proba_cdtcdrn))

#confmat_test:

#confmat_test:
# [[42630  7829]
# [  993  2520]]
#the acc is: 0.8365448751204329
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.84      0.91     50459
#        1.0       0.24      0.72      0.36      3513
#
#avg / total       0.93      0.84      0.87     53972
#
#the auc of dt is: 0.793978059100351
#the brier socre is 0.15167961914925823
###########################################################################
#cdrn12
dfcdrn12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew12_2.csv', delimiter=',', header=0)
dfcdrn12 = pd.merge(dfcdrn12,visitexafter1y,on = ['person_id','condition_start_date'], how='inner')
df2cdrn12 = pd.DataFrame(dfcdrn12, columns=columnsdt2)

df2cdrn12[np.isnan(df2cdrn12)] = '0'
ycdrn12 = df2cdrn12['ppd']
Xcdrn12 = df2cdrn12.drop(columns=['ppd'])

clfDecisionTree.fit(X_train_sfsdt, y_traindt)
y_pred_ccdrn = clfDecisionTree.predict(Xcdrn12)
y_pred_proba_cdtcdrn = clfDecisionTree.predict_proba(Xcdrn12)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn12, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn12, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn12, y_pred_ccdrn))
print('the auc of dt is:', roc_auc_score(ycdrn12, y_pred_proba_cdtcdrn))
print('the brier socre is',brier_score_loss(ycdrn12, y_pred_proba_cdtcdrn))

#confmat_test:
# [[42676  7783]
# [ 1022  2491]]
#the acc is: 0.8368598532572445
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.85      0.91     50459
#        1.0       0.24      0.71      0.36      3513
#
#avg / total       0.93      0.84      0.87     53972
#
#the auc of dt is: 0.7895224712177791
#the brier socre is 0.15153428420862264

