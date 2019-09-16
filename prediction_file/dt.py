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
df = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodelwithhistory.csv', delimiter=',', header=0)
df['ppd'].value_counts()
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visitexafter1ywcmnew.csv', sep=',')
visitexafter1y = pd.DataFrame(visitexafter1y,columns=['person_id','condition_start_date'])
df_2 = pd.merge(df,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df_2['ppd'].value_counts()

#withouthistroy
clfDecisionTree = tree.DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=8,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=230,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=1, splitter='best')


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

columnsdt_2 = pd.DataFrame(columnsdt, columns=['newvariable']) #change list to dataframe
#variable name
name_list = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/name_list.csv',delimiter=',', header=0)
columnsnamedt = pd.merge(columnsdt_2, name_list, on = ['newvariable'], how='left')
columnsnamedt.to_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/result/selectedfeature1908/dtfeaturelist.csv',header=0)

df3 = pd.DataFrame(df_2, columns=columnsdt)
df3_2 = df3
df3_2['a'] = 1
y = df3_2['ppd']
X = df3_2.drop(columns=['ppd'])
X_train_sfsdt, X_test_sfsdt, y_traindt, y_testdt = train_test_split(X, y, test_size=0.2, random_state=17)

dttrain = pd.concat([X_train_sfsdt, y_traindt],axis=1)
for i in columnsdt:
    print(dttrain.groupby(['ppd',i])['a'].count())


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
# [[2684  149]
# [  14   72]]
#the acc is: 0.9441589585474478
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.95      0.97      2833
#        1.0       0.33      0.84      0.47        86
#
#avg / total       0.98      0.94      0.96      2919
#
#the auc of random forest is: 0.9058316026235644
#the brier socre is 0.07878689676595187
import pickle 
with open('/Users/bu/Desktop/clfdecisiontree.pickle', 'wb') as f:
    pickle.dump(clfDecisionTree, f)
with open('/Users/bu/Desktop/clfdecisiontree.pickle', 'rb') as f:
    clf2 = pickle.load(f)
    print(clf2.predict(X[0:1]))
###########################################################################
#wcmmodel30
dfwcmmodel30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel30.csv', delimiter=',', header=0)
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
# [[2716  117]
# [  17   69]]
#the acc is: 0.9540938677629325
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.96      0.98      2833
#        1.0       0.37      0.80      0.51        86
#
#avg / total       0.98      0.95      0.96      2919
#
#the auc of random forest is: 0.8959850269662367
#the brier socre is 0.06910725586927087

###########################################################################
#wcmmodel24
dfwcmmodel24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel24.csv', delimiter=',', header=0)
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
# [[2715  118]
# [  17   69]]
#the acc is: 0.9537512846865365
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.96      0.98      2833
#        1.0       0.37      0.80      0.51        86
#
#avg / total       0.98      0.95      0.96      2919
#
#the auc of random forest is: 0.8938917574434202
#the brier socre is 0.06795367471883221

###########################################################################
#wcmmodel18
dfwcmmodel18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel18.csv', delimiter=',', header=0)
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
# [[2746   87]
# [  17   69]]
#the acc is: 0.9643713600548133
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2833
#        1.0       0.44      0.80      0.57        86
#
#avg / total       0.98      0.96      0.97      2919
#
#the auc of random forest is: 0.8972081530795689
#the brier socre is 0.06699072697492502

###########################################################################
#wcmmodel12
dfwcmmodel12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel12.csv', delimiter=',', header=0)
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
# [[2749   84]
# [  17   69]]
#the acc is: 0.9653991092840014
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2833
#        1.0       0.45      0.80      0.58        86
#
#avg / total       0.98      0.97      0.97      2919
#
#the auc of random forest is: 0.8896662261223619
#the brier socre is 0.04231583709743831
###########################################################################
#cdrn
dfcdrn = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
dfcdrn = pd.merge(dfcdrn,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
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
# [[42435  7989]
# [  214  1675]]
#the acc is: 0.8431938523885076
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.84      0.91     50424
#        1.0       0.17      0.89      0.29      1889
#
#avg / total       0.97      0.84      0.89     52313
#
#the auc of logistics is: 0.8770158332092399
#the brier socre is 0.15229559311751167
###########################################################################
#cdrn30
dfcdrn30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew30.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
dfcdrn30 = pd.merge(dfcdrn30,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
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

#fit without history
#confmat_test:
# [[42562  7862]
# [  226  1663]]
#the acc is: 0.845392158736834
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.84      0.91     50424
#        1.0       0.17      0.88      0.29      1889
#
#avg / total       0.97      0.85      0.89     52313
#
#the auc of logistics is: 0.8744692230635928
#the brier socre is 0.15070766523777962

###########################################################################
#cdrn24
dfcdrn24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew24.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
dfcdrn24 = pd.merge(dfcdrn24,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
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
# [[42637  7787]
# [  228  1661]]
#the acc is: 0.8467876053753369
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.85      0.91     50424
#        1.0       0.18      0.88      0.29      1889
#
#avg / total       0.97      0.85      0.89     52313
#
#the auc of dt is: 0.8740073115921927
#the brier socre is 0.14977640255216204
###########################################################################
#cdrn18
dfcdrn18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew18.csv', delimiter=',', header=0)
dfcdrn18 = pd.merge(dfcdrn18,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
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
# [[42692  7732]
# [  234  1655]]
#the acc is: 0.8477242750367977
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.85      0.91     50424
#        1.0       0.18      0.88      0.29      1889
#
#avg / total       0.97      0.85      0.89     52313
#
#the auc of dt is: 0.8726630098417091
#the brier socre is 0.15438652518827065

#confmat_test:
# [[42692  7732]
# [  234  1655]]
#the acc is: 0.8477242750367977
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.85      0.91     50424
#        1.0       0.18      0.88      0.29      1889
#
#avg / total       0.97      0.85      0.89     52313
#
#the auc of dt is: 0.87266184449883
#the brier socre is 0.14913405340994124
###########################################################################
#cdrn12
dfcdrn12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew12.csv', delimiter=',', header=0)
dfcdrn12 = pd.merge(dfcdrn12,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
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
# [[42740  7684]
# [  239  1650]]
#the acc is: 0.848546250453998
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.85      0.92     50424
#        1.0       0.18      0.87      0.29      1889
#
#avg / total       0.96      0.85      0.89     52313
#
#the auc of dt is: 0.8716471563072095
#the brier socre is 0.14850290366915442



