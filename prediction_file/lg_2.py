#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:17:50 2019

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
pd.set_option('display.max_columns', None)

###########################################################################
df = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodelwithhistory_2.csv', delimiter=',', header=0)
df['ppd'].value_counts()
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visitexafter1ywcmnew_2.csv', sep=',')
visitexafter1y = pd.DataFrame(visitexafter1y,columns=['person_id','condition_start_date'])
df_2 = pd.merge(df,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df_2['ppd'].value_counts()
#withouthistroy
clfLogisticRegression = LogisticRegression(C=0.009, class_weight='balanced', dual=False,
                   fit_intercept=True, intercept_scaling=1,
                   max_iter=100, n_jobs=None, penalty='l2',
                   random_state=None, tol=0.0001, verbose=0,
                   warm_start=False)


columnslg25 = ['ppd','edvisitcount', 'white', 'asian', 'single', 'C11.0', 'C1344.0', 'C1387.0', 'C1685.0', 'C1809.0', 'C2057.0', 'C2359.0', 'C236.0', 'C2398.0', 'C2903.0', 'C3030.0', 'C3059.0', 'C3066.0', 'C3473.0', 'C3830.0', 'C419.0', 'C4527.0', 'M14.0', 'M21.0', 'M39.0', 'M88.0']

##withhistory
#clfLogisticRegression = LogisticRegression(C=0.09, class_weight='balanced', dual=False,
#                   fit_intercept=True, intercept_scaling=1,
#                   max_iter=100, n_jobs=None, penalty='l1',
#                   random_state=None, tol=0.0001, verbose=0,
#                   warm_start=False)


columnslg19 = ['ppd','dbp3rd', 'white', 'C137.0', 'C1809.0', 'C2043.0', 'C2214.0', 'C2544.0', 'C2644.0', 'C345.0', 'C3626.0', 'C3725.0', 'C419.0', 'C4879.0', 'C83.0', 'M14.0', 'M54.0', 'anxiety', 'mooddisorder', 'otherdisorder']

###########################################################################
columnslg = columnslg19 + columnslg25
columnslg = list(dict.fromkeys(columnslg))
#wcm
df3 = pd.DataFrame(df_2, columns=columnslg)
del df3['C137.0']
del df3['C2544.0']
del df3['C2644.0']

def abdominal(a, b):
	if a == 1 or b == 1:
		return 1
	else:
		return 0
#print(frame, '\n')
df3['abdominalpain'] = df3.apply(lambda x: abdominal(x['C11.0'], x['C2043.0']), axis = 1)


def primigravida(a, b):
	if a == 1 or b == 1:
		return 1
	else:
		return 0
#print(frame, '\n')
df3['primigravida'] = df3.apply(lambda x: primigravida(x['C1685.0'], x['C3830.0']), axis = 1)


def vomit(a, b):
	if a == 1 or b == 1:
		return 1
	else:
		return 0
#print(frame, '\n')
df3['vomit'] = df3.apply(lambda x: vomit(x['C4879.0'], x['C3066.0']), axis = 1)


df3 = df3.drop(['C11.0', 'C2043.0', 'C1685.0', 'C3830.0', 'C4879.0','C3066.0','C345.0','C2057.0'], axis=1)
df3['ppd'].value_counts()
#0.0    14187
#1.0     1010

y = df3['ppd']
X = df3.drop(columns=['ppd'])
X_train_sfslg, X_test_sfslg, y_trainlg, y_testlg = train_test_split(X, y, test_size=0.2, random_state=17)
print(X_test_sfslg.shape)
print(X_train_sfslg.shape)

#dftrain = pd.concat([X_train_sfslg, y_trainlg],axis=1)
#dftrain['ppd'].value_counts()
#
#dftest = pd.concat([X_test_sfslg, y_testlg],axis=1)
#dftest['ppd'].value_counts()

#print(dftrain.groupby(['ppd'])['dbp3rd','edvisitcount'].mean())
#print(dftrain.groupby(['ppd'])['dbp3rd','edvisitcount'].std())

clfLogisticRegression.fit(X_train_sfslg, y_trainlg)
y_pred_c = clfLogisticRegression.predict(X_test_sfslg)
y_pred_proba_clg = clfLogisticRegression.predict_proba(X_test_sfslg)[:, 1]
confmat_test_c = confusion_matrix(y_true=y_testlg, y_pred=y_pred_c)

print('confmat_test:\n', confmat_test_c)
print('the acc is:', accuracy_score(y_testlg, y_pred_c))
print('the classification_report:', classification_report(y_testlg, y_pred_c))
print('the auc of logistics is:', roc_auc_score(y_testlg, y_pred_proba_clg))
print('the brier socre is',brier_score_loss(y_testlg, y_pred_proba_clg))

#fit lg withouthistory
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
#the auc of logistics is: 0.9367440814127649
#the brier socre is 0.08243278092493153
import pickle #pickle模块
#保存Model(注:save文件夹要预先建立，否则会报错)
with open('/Users/bu/Desktop/clflogistics_2.pickle', 'wb') as f:
    pickle.dump(clfLogisticRegression, f)

###########################################################################
#wcmmodel30
dfwcmmodel30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel30_2.csv', delimiter=',', header=0)
dfwcmmodel30 = pd.merge(dfwcmmodel30,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
columnslg = columnslg19 + columnslg25
columnslg = list(dict.fromkeys(columnslg))
columnslg.remove('C137.0')
columnslg.remove('C2544.0')
columnslg.remove('C2644.0')
df2wcmmodel30 = pd.DataFrame(dfwcmmodel30, columns=columnslg)
df2wcmmodel30['abdominalpain'] = df2wcmmodel30.apply(lambda x: abdominal(x['C11.0'], x['C2043.0']), axis = 1)
df2wcmmodel30['primigravida'] = df2wcmmodel30.apply(lambda x: primigravida(x['C1685.0'], x['C3830.0']), axis = 1)
df2wcmmodel30['vomit'] = df2wcmmodel30.apply(lambda x: vomit(x['C4879.0'], x['C3066.0']), axis = 1)
df2wcmmodel30 = df2wcmmodel30.drop(['C11.0', 'C2043.0', 'C1685.0', 'C3830.0', 'C4879.0','C3066.0','C345.0','C2057.0'], axis=1)



df2wcmmodel30[np.isnan(df2wcmmodel30)] = '0'
y30 = df2wcmmodel30['ppd']
X30 = df2wcmmodel30.drop(columns=['ppd'])

X_train30, X_test30, y_train30, y_test30 = train_test_split(X30, y30, test_size=0.2, random_state=17)

clfLogisticRegression.fit(X_train_sfslg, y_trainlg)
y_pred_cwcmmodel30 = clfLogisticRegression.predict(X_test30)
y_pred_proba_clgwcmmodel30 = clfLogisticRegression.predict_proba(X_test30)[:, 1]
confmat_test_cwcmmodel30 = confusion_matrix(y_true=y_test30, y_pred=y_pred_cwcmmodel30)

print('confmat_test:\n', confmat_test_cwcmmodel30)
print('the acc is:', accuracy_score(y_test30, y_pred_cwcmmodel30))
print('the classification_report:', classification_report(y_test30, y_pred_cwcmmodel30))
print('the auc of logistics is:', roc_auc_score(y_test30, y_pred_proba_clgwcmmodel30))
print('the brier socre is',brier_score_loss(y_test30, y_pred_proba_clgwcmmodel30))

#fit lg withouthistory
#confmat_test:
# [[2775   88]
# [  37  140]]
#the acc is: 0.9588815789473685
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2863
#        1.0       0.61      0.79      0.69       177
#
#avg / total       0.97      0.96      0.96      3040
#
#the auc of logistics is: 0.9211338507472111
#the brier socre is 0.07425621870199521



###########################################################################
#wcmmodel24
dfwcmmodel24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel24_2.csv', delimiter=',', header=0)
dfwcmmodel24 = pd.merge(dfwcmmodel24,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel24 = pd.DataFrame(dfwcmmodel24, columns=columnslg)
df2wcmmodel24['abdominalpain'] = df2wcmmodel24.apply(lambda x: abdominal(x['C11.0'], x['C2043.0']), axis = 1)
df2wcmmodel24['primigravida'] = df2wcmmodel24.apply(lambda x: primigravida(x['C1685.0'], x['C3830.0']), axis = 1)
df2wcmmodel24['vomit'] = df2wcmmodel24.apply(lambda x: vomit(x['C4879.0'], x['C3066.0']), axis = 1)
df2wcmmodel24 = df2wcmmodel24.drop(['C11.0', 'C2043.0', 'C1685.0', 'C3830.0', 'C4879.0','C3066.0','C345.0','C2057.0'], axis=1)

df2wcmmodel24[np.isnan(df2wcmmodel24)] = '0'
y24 = df2wcmmodel24['ppd']
X24 = df2wcmmodel24.drop(columns=['ppd'])

X_train24, X_test24, y_train24, y_test24 = train_test_split(X24, y24, test_size=0.2, random_state=17)

clfLogisticRegression.fit(X_train_sfslg, y_trainlg)
y_pred_cwcmmodel24 = clfLogisticRegression.predict(X_test24)
y_pred_proba_clgwcmmodel24 = clfLogisticRegression.predict_proba(X_test24)[:, 1]
confmat_test_cwcmmodel24 = confusion_matrix(y_true=y_test24, y_pred=y_pred_cwcmmodel24)

print('confmat_test:\n', confmat_test_cwcmmodel24)
print('the acc is:', accuracy_score(y_test24, y_pred_cwcmmodel24))
print('the classification_report:', classification_report(y_test24, y_pred_cwcmmodel24))
print('the auc of logistics is:', roc_auc_score(y_test24, y_pred_proba_clgwcmmodel24))
print('the brier socre is',brier_score_loss(y_test24, y_pred_proba_clgwcmmodel24))

#without history
#confmat_test:
# [[2775   88]
# [  37  140]]
#the acc is: 0.9588815789473685
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2863
#        1.0       0.61      0.79      0.69       177
#
#avg / total       0.97      0.96      0.96      3040
#
#the auc of logistics is: 0.9221017817429074
#the brier socre is 0.0740982129003969


###########################################################################
#wcmmodel18
dfwcmmodel18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel18_2.csv', delimiter=',', header=0)
dfwcmmodel18 = pd.merge(dfwcmmodel18,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel18 = pd.DataFrame(dfwcmmodel18, columns=columnslg)
df2wcmmodel18['abdominalpain'] = df2wcmmodel18.apply(lambda x: abdominal(x['C11.0'], x['C2043.0']), axis = 1)
df2wcmmodel18['primigravida'] = df2wcmmodel18.apply(lambda x: primigravida(x['C1685.0'], x['C3830.0']), axis = 1)
df2wcmmodel18['vomit'] = df2wcmmodel18.apply(lambda x: vomit(x['C4879.0'], x['C3066.0']), axis = 1)
df2wcmmodel18 = df2wcmmodel18.drop(['C11.0', 'C2043.0', 'C1685.0', 'C3830.0', 'C4879.0','C3066.0','C345.0','C2057.0'], axis=1)

df2wcmmodel18[np.isnan(df2wcmmodel18)] = '0'
y18 = df2wcmmodel18['ppd']
X18 = df2wcmmodel18.drop(columns=['ppd'])
X_train18, X_test18, y_train18, y_test18 = train_test_split(X18, y18, test_size=0.2, random_state=17)

clfLogisticRegression.fit(X_train_sfslg, y_trainlg)
y_pred_cwcmmodel18 = clfLogisticRegression.predict(X_test18)
y_pred_proba_clgwcmmodel18 = clfLogisticRegression.predict_proba(X_test18)[:, 1]
confmat_test_cwcmmodel18 = confusion_matrix(y_true=y_test18, y_pred=y_pred_cwcmmodel18)

print('confmat_test:\n', confmat_test_cwcmmodel18)
print('the acc is:', accuracy_score(y_test18, y_pred_cwcmmodel18))
print('the classification_report:', classification_report(y_test18, y_pred_cwcmmodel18))
print('the auc of logistics is:', roc_auc_score(y_test18, y_pred_proba_clgwcmmodel18))
print('the brier socre is',brier_score_loss(y_test18, y_pred_proba_clgwcmmodel18))

#withouthistory
#confmat_test:
# [[2773   90]
# [  37  140]]
#the acc is: 0.9582236842105263
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2863
#        1.0       0.61      0.79      0.69       177
#
#avg / total       0.96      0.96      0.96      3040
#
#the auc of logistics is: 0.9191052410355383
#the brier socre is 0.07413180802441073


###########################################################################
#wcmmodel12
dfwcmmodel12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel12_2.csv', delimiter=',', header=0)
dfwcmmodel12 = pd.merge(dfwcmmodel12,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel12 = pd.DataFrame(dfwcmmodel12, columns=columnslg)
df2wcmmodel12['abdominalpain'] = df2wcmmodel12.apply(lambda x: abdominal(x['C11.0'], x['C2043.0']), axis = 1)
df2wcmmodel12['primigravida'] = df2wcmmodel12.apply(lambda x: primigravida(x['C1685.0'], x['C3830.0']), axis = 1)
df2wcmmodel12['vomit'] = df2wcmmodel12.apply(lambda x: vomit(x['C4879.0'], x['C3066.0']), axis = 1)
df2wcmmodel12 = df2wcmmodel12.drop(['C11.0', 'C2043.0', 'C1685.0', 'C3830.0', 'C4879.0','C3066.0','C345.0','C2057.0'], axis=1)

df2wcmmodel12[np.isnan(df2wcmmodel12)] = '0'
y12 = df2wcmmodel12['ppd']
X12 = df2wcmmodel12.drop(columns=['ppd'])

X_train12, X_test12, y_train12, y_test12 = train_test_split(X12, y12, test_size=0.2, random_state=17)

clfLogisticRegression.fit(X_train_sfslg, y_trainlg)
y_pred_cwcmmodel12 = clfLogisticRegression.predict(X_test12)
y_pred_proba_clgwcmmodel12 = clfLogisticRegression.predict_proba(X_test12)[:, 1]
confmat_test_cwcmmodel12 = confusion_matrix(y_true=y_test12, y_pred=y_pred_cwcmmodel12)

print('confmat_test:\n', confmat_test_cwcmmodel12)
print('the acc is:', accuracy_score(y_test12, y_pred_cwcmmodel12))
print('the classification_report:', classification_report(y_test12, y_pred_cwcmmodel12))
print('the auc of logistics is:', roc_auc_score(y_test12, y_pred_proba_clgwcmmodel12))
print('the brier socre is',brier_score_loss(y_test12, y_pred_proba_clgwcmmodel12))

#without history
#confmat_test:
# [[2775   88]
# [  37  140]]
#the acc is: 0.9588815789473685
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2863
#        1.0       0.61      0.79      0.69       177
#
#avg / total       0.97      0.96      0.96      3040
#
#the auc of logistics is: 0.920642485165298
#the brier socre is 0.07393930955275153


###########################################################################
#cdrn
columnslg = columnslg19 + columnslg25
columnslg = list(dict.fromkeys(columnslg))
columnslg.remove('C137.0')
columnslg.remove('C2544.0')
columnslg.remove('C2644.0')

dfcdrn = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew_2.csv', delimiter=',', header=0)
dfcdrn['ppd'].value_counts()
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew_2.csv', delimiter=',', header=0)
dfcdrn = pd.merge(dfcdrn,visitexafter1y,on = ['person_id','condition_start_date'], how='inner')
#one_hotrace = pd.get_dummies(dfcdrn['race2'])
##1:white
##2:asian
##3:black
##4:combine
##5:other
#one_hotrace = one_hotrace.rename({1: 'white',2:'asian',3:'black',4:'combine',5:'otherrace'}, axis=1) #'a': 'X', 'b': 'Y'
#dfcdrn = dfcdrn.drop('race2',axis = 1)
#dfcdrn = dfcdrn.join(one_hotrace)

df2cdrn = pd.DataFrame(dfcdrn, columns=columnslg)
df2cdrn['abdominalpain'] = df2cdrn.apply(lambda x: abdominal(x['C11.0'], x['C2043.0']), axis = 1)
df2cdrn['primigravida'] = df2cdrn.apply(lambda x: primigravida(x['C1685.0'], x['C3830.0']), axis = 1)
df2cdrn['vomit'] = df2cdrn.apply(lambda x: vomit(x['C4879.0'], x['C3066.0']), axis = 1)
df2cdrn = df2cdrn.drop(['C11.0', 'C2043.0', 'C1685.0', 'C3830.0', 'C4879.0','C3066.0','C345.0','C2057.0'], axis=1)

df2cdrn['ppd'].value_counts()
#0.0    50459
#1.0     3513

df2cdrn[np.isnan(df2cdrn)] = '0'
ycdrn = df2cdrn['ppd']
Xcdrn = df2cdrn.drop(columns=['ppd'])

new_names=['ppd',
'Diastolicbloodpressureinthirdtrimester',
'White',
'Falselabor',
'Hemorrhage',
'Placentalinfarct',
'Pre-eclampsia',
'Anxietyinpregnancy',
'Abnormalityoforgans',
'Antidepressants',
'Directactingantivirals',
'Anxietyhistory',
'Mooddisorderhistory',
'Otherdisorderhistory',
'#EDvisit',
'Asian',
'Single',
'Deliveriesbycesarean',
'Diarrhea',
'Hypertensivedisorder',
'Acutepharyngitis',
'Hypothyroidism',
'Depressionin',
'Mentaldisorderin',
'Migraine',
'Palpitations',
'Threatenedmiscarriage',
'Antihistamines',
'Betablockingagents',
'Otherantibacterials',
'Abdominalpain',
'Primigravida',
'Vomitingin'
]
feature_names = df2cdrn.columns.values.tolist()
df2cdrn.rename(columns=dict(zip(feature_names, new_names)), inplace=True)


#df2cdrn.to_csv('/Users/bu/Desktop/df2cdrn.csv', sep=',',index=0)

clfLogisticRegression.fit(X_train_sfslg, y_trainlg)
y_pred_ccdrn = clfLogisticRegression.predict(Xcdrn)
y_pred_proba_clgcdrn = clfLogisticRegression.predict_proba(Xcdrn)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn, y_pred_ccdrn))
print('the auc of logistics is:', roc_auc_score(ycdrn, y_pred_proba_clgcdrn))
print('the brier socre is',brier_score_loss(ycdrn, y_pred_proba_clgcdrn))

#withouthistory
#confmat_test:
# [[42215  8244]
# [  687  2826]]
#the acc is: 0.8345253094196992
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.84      0.90     50459
#        1.0       0.26      0.80      0.39      3513
#
#avg / total       0.94      0.83      0.87     53972
#
#the auc of logistics is: 0.8857082249678947
#the brier socre is 0.1581337923892438

###########################################################################
#cdrn30
dfcdrn30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew30_2.csv', delimiter=',', header=0)
dfcdrn30 = pd.merge(dfcdrn30,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn30 = pd.DataFrame(dfcdrn30, columns=columnslg)
df2cdrn30['abdominalpain'] = df2cdrn30.apply(lambda x: abdominal(x['C11.0'], x['C2043.0']), axis = 1)
df2cdrn30['primigravida'] = df2cdrn30.apply(lambda x: primigravida(x['C1685.0'], x['C3830.0']), axis = 1)
df2cdrn30['vomit'] = df2cdrn30.apply(lambda x: vomit(x['C4879.0'], x['C3066.0']), axis = 1)
df2cdrn30 = df2cdrn30.drop(['C11.0', 'C2043.0', 'C1685.0', 'C3830.0', 'C4879.0','C3066.0','C345.0','C2057.0'], axis=1)


df2cdrn30[np.isnan(df2cdrn30)] = '0'
ycdrn30 = df2cdrn30['ppd']
Xcdrn30 = df2cdrn30.drop(columns=['ppd'])

clfLogisticRegression.fit(X_train_sfslg, y_trainlg)
y_pred_ccdrn = clfLogisticRegression.predict(Xcdrn30)
y_pred_proba_clgcdrn = clfLogisticRegression.predict_proba(Xcdrn30)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn30, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn30, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn30, y_pred_ccdrn))
print('the auc of logistics is:', roc_auc_score(ycdrn30, y_pred_proba_clgcdrn))
print('the brier socre is',brier_score_loss(ycdrn30, y_pred_proba_clgcdrn))

#fit lg withouthistory
#confmat_test:
# [[42712  7747]
# [ 1001  2512]]
#the acc is: 0.8379159564218484
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.85      0.91     50459
#        1.0       0.24      0.72      0.36      3513
#
#avg / total       0.93      0.84      0.87     53972
#
#the auc of logistics is: 0.8240975400618791
#the brier socre is 0.15368232551394712

###########################################################################
#cdrn24
dfcdrn24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew24_2.csv', delimiter=',', header=0)
dfcdrn24 = pd.merge(dfcdrn24,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn24 = pd.DataFrame(dfcdrn24, columns=columnslg)
df2cdrn24['abdominalpain'] = df2cdrn24.apply(lambda x: abdominal(x['C11.0'], x['C2043.0']), axis = 1)
df2cdrn24['primigravida'] = df2cdrn24.apply(lambda x: primigravida(x['C1685.0'], x['C3830.0']), axis = 1)
df2cdrn24['vomit'] = df2cdrn24.apply(lambda x: vomit(x['C4879.0'], x['C3066.0']), axis = 1)
df2cdrn24 = df2cdrn24.drop(['C11.0', 'C2043.0', 'C1685.0', 'C3830.0', 'C4879.0','C3066.0','C345.0','C2057.0'], axis=1)

df2cdrn24[np.isnan(df2cdrn24)] = '0'
ycdrn24 = df2cdrn24['ppd']
Xcdrn24 = df2cdrn24.drop(columns=['ppd'])

clfLogisticRegression.fit(X_train_sfslg, y_trainlg)
y_pred_ccdrn = clfLogisticRegression.predict(Xcdrn24)
y_pred_proba_clgcdrn = clfLogisticRegression.predict_proba(Xcdrn24)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn24, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn24, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn24, y_pred_ccdrn))
print('the auc of logistics is:', roc_auc_score(ycdrn24, y_pred_proba_clgcdrn))
print('the brier socre is',brier_score_loss(ycdrn24, y_pred_proba_clgcdrn))

#confmat_test:
# [[42783  7676]
# [ 1018  2495]]
#the acc is: 0.8389164752093679
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.85      0.91     50459
#        1.0       0.25      0.71      0.36      3513
#
#avg / total       0.93      0.84      0.87     53972
#
#the auc of logistics is: 0.821381527991484
#the brier socre is 0.15237603500025138

###########################################################################
#cdrn18
dfcdrn18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew18_2.csv', delimiter=',', header=0)
dfcdrn18 = pd.merge(dfcdrn18,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn18 = pd.DataFrame(dfcdrn18, columns=columnslg)
df2cdrn18['abdominalpain'] = df2cdrn18.apply(lambda x: abdominal(x['C11.0'], x['C2043.0']), axis = 1)
df2cdrn18['primigravida'] = df2cdrn18.apply(lambda x: primigravida(x['C1685.0'], x['C3830.0']), axis = 1)
df2cdrn18['vomit'] = df2cdrn18.apply(lambda x: vomit(x['C4879.0'], x['C3066.0']), axis = 1)
df2cdrn18 = df2cdrn18.drop(['C11.0', 'C2043.0', 'C1685.0', 'C3830.0', 'C4879.0','C3066.0','C345.0','C2057.0'], axis=1)

df2cdrn18[np.isnan(df2cdrn18)] = '0'
ycdrn18 = df2cdrn18['ppd']
Xcdrn18 = df2cdrn18.drop(columns=['ppd'])

clfLogisticRegression.fit(X_train_sfslg, y_trainlg)
y_pred_ccdrn = clfLogisticRegression.predict(Xcdrn18)
y_pred_proba_clgcdrn = clfLogisticRegression.predict_proba(Xcdrn18)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn18, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn18, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn18, y_pred_ccdrn))
print('the auc of logistics is:', roc_auc_score(ycdrn18, y_pred_proba_clgcdrn))
print('the brier socre is',brier_score_loss(ycdrn18, y_pred_proba_clgcdrn))
#without history
#confmat_test:
# [[42821  7638]
# [ 1041  2472]]
#the acc is: 0.8391943970947899
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.85      0.91     50459
#        1.0       0.24      0.70      0.36      3513
#
#avg / total       0.93      0.84      0.87     53972
#
#the auc of logistics is: 0.8165126179813349
#the brier socre is 0.15138511882868264

###########################################################################
#cdrn12
dfcdrn12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew12_2.csv', delimiter=',', header=0)
dfcdrn12 = pd.merge(dfcdrn12,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn12 = pd.DataFrame(dfcdrn12, columns=columnslg)
df2cdrn12['abdominalpain'] = df2cdrn12.apply(lambda x: abdominal(x['C11.0'], x['C2043.0']), axis = 1)
df2cdrn12['primigravida'] = df2cdrn12.apply(lambda x: primigravida(x['C1685.0'], x['C3830.0']), axis = 1)
df2cdrn12['vomit'] = df2cdrn12.apply(lambda x: vomit(x['C4879.0'], x['C3066.0']), axis = 1)
df2cdrn12 = df2cdrn12.drop(['C11.0', 'C2043.0', 'C1685.0', 'C3830.0', 'C4879.0','C3066.0','C345.0','C2057.0'], axis=1)

df2cdrn12[np.isnan(df2cdrn12)] = '0'
ycdrn12 = df2cdrn12['ppd']
Xcdrn12 = df2cdrn12.drop(columns=['ppd'])

clfLogisticRegression.fit(X_train_sfslg, y_trainlg)
y_pred_ccdrn = clfLogisticRegression.predict(Xcdrn12)
y_pred_proba_clgcdrn = clfLogisticRegression.predict_proba(Xcdrn12)[:, 1]
confmat_test_ccdrn = confusion_matrix(y_true=ycdrn12, y_pred=y_pred_ccdrn)

print('confmat_test:\n', confmat_test_ccdrn)
print('the acc is:', accuracy_score(ycdrn12, y_pred_ccdrn))
print('the classification_report:', classification_report(ycdrn12, y_pred_ccdrn))
print('the auc of logistics is:', roc_auc_score(ycdrn12, y_pred_proba_clgcdrn))
print('the brier socre is',brier_score_loss(ycdrn12, y_pred_proba_clgcdrn))

#withouthistory
#confmat_test:
# [[42853  7606]
# [ 1059  2454]]
#the acc is: 0.8394537908545172
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.85      0.91     50459
#        1.0       0.24      0.70      0.36      3513
#
#avg / total       0.93      0.84      0.87     53972
#
#the auc of logistics is: 0.8104601127998519
#the brier socre is 0.1499402752302699
