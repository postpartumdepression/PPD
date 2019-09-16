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
df = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodelwithhistory.csv', delimiter=',', header=0)
df['ppd'].value_counts()
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visitexafter1ywcmnew.csv', sep=',')
visitexafter1y = pd.DataFrame(visitexafter1y,columns=['person_id','condition_start_date'])
df_2 = pd.merge(df,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df_2['ppd'].value_counts()
#0.0    14178
#1.0      416
#withouthistroy
clfLogisticRegression = LogisticRegression(C=0.009, class_weight='balanced', dual=False,
                   fit_intercept=True, intercept_scaling=1,
                   max_iter=100, n_jobs=None, penalty='l2',
                   random_state=None, tol=0.0001, verbose=0,
                   warm_start=False)


columnslg25 = ['ppd','edvisitcount', 'white', 'asian', 'single', 'C11.0', 'C1344.0', 'C1387.0', 'C1685.0', 'C1809.0', 'C2057.0', 'C2359.0', 'C236.0', 'C2398.0', 'C2903.0', 'C3030.0', 'C3059.0', 'C3066.0', 'C3473.0', 'C3830.0', 'C419.0', 'C4527.0', 'M14.0', 'M21.0', 'M39.0', 'M88.0']

#withhistory
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

##########################################################
condition_list = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/condition_listcombine.csv',delimiter=',', header=0)
medication_list = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/medication_listnew.csv',delimiter=',', header=0)
medication_list = medication_list.rename({'concept_name.1': 'concept_name_y'}, axis=1) #'a': 'X', 'b': 'Y'
name_list = pd.concat([medication_list,condition_list],axis = 0)
name_list.to_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/name_list.csv',header=1)


feature_names = df3.columns.values.tolist()#[2:]
columnsori = feature_names
columns = pd.DataFrame({'newvariable': columnsori})


columns2 = pd.merge(columns, name_list, on = ['newvariable'], how='left')
#columns2.loc[columns2['concept_name'].isnull(),'concept_name']=columns2[columns2['concept_name'].isnull()]['newvariable']
columns2.to_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/result/selectedfeature1909/lgfeaturelist.csv',header=0)
#remove duplicate name in namelist
columns2 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/result/selectedfeature1909/lgfeaturelist.csv',delimiter=',', header=0)

##########################################################

df3 = pd.DataFrame(df_2, columns=columnslg)

#OR
def abdominal(a, b):
	if a == 1 or b == 1:
		return 1
	else:
		return 0
#print(frame, '\n')
df3['abdominalpain'] = df3.apply(lambda x: abdominal(x['C11.0'], x['C2043.0']), axis = 1)

#df3.groupby(['C1364.0'])['ppd'].count()#37
#df3.groupby(['C2917.0'])['ppd'].count()#65
#df3.groupby(['depressionpregnancy'])['ppd'].count()#82

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
del df3['C137.0']
del df3['C2544.0']
del df3['C2644.0']

#rename columns name
feature_names = df3.columns.values.tolist()
columns = pd.DataFrame({'newvariable': feature_names})
name_list = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/name_list.csv',delimiter=',', header=0)

columns2  = pd.merge(columns, name_list, on = ['newvariable'], how='left')
columns2.loc[columns2['concept_name_y'].isnull(),'concept_name_y']=columns2[columns2['concept_name_y'].isnull()]['newvariable']
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


df3.rename(columns=dict(zip(feature_names, new_names)), inplace=True)
df3['a'] = 1
y3 = df3['ppd']
X3 = df3.drop(columns=['ppd'])
X_train_sfslg3, X_test_sfslg3, y_trainlg3, y_testlg3 = train_test_split(X3, y3, test_size=0.2, random_state=17)
dftrain = pd.concat([X_train_sfslg3, y_trainlg3],axis=1)
for i in new_names:
    print(dftrain.groupby(['ppd',i])['a'].count())
    
# dftest = pd.concat([X_test_sfslg3, y_testlg3],axis=1)
# dftest['ppd'].value_counts()

df3.to_csv('/Users/bu/Desktop/df3.csv',header=1)
dftrain.to_csv('/Users/bu/Desktop/dftrain.csv',header=1)



new_names2 = ['PPD',
'Diastolic blood pressure in third trimester',
'White',
'False labor at or after 37 completed weeks of gestation',
'Hemorrhage in early pregnancy, antepartum',
'Placental infarct',
'Pre-eclampsia',
'Anxiety in pregnancy',
'Abnormality of organs AND/OR soft tissues of pelvis affecting pregnancy',
'Antidepressants',
'Direct acting antivirals',
'Anxiety history',
'Mood disorder history',
'Other disorder history',
'#ED visit',
'Asian',
'Single',
'Deliveries by cesarean',
'Diarrhea',
'Hypertensive disorder',
'Acute pharyngitis',
'Hypothyroidism',
'Depression in pregnancy',
'Mental disorder in pregnancy',
'Migraine',
'Palpitations',
'Threatened miscarriage',
'Antihistamines for systemic use',
'Beta blocking agents',
'Other antibacterials',
'Abdominal pain',
'Primigravida',
'Vomiting in pregnancy'
]
df4=df3
df4.rename(columns=dict(zip(new_names, new_names2)), inplace=True)
del df4['a']

#############corr
import seaborn as sns
sns.set(font_scale=5)
plt.figure(figsize=(50,50)) 
# play with the figsize until the plot is big enough to plot all the columns
# of your dataset, or the way you desire it to look like otherwise

#
#df4 = df4.drop(columns=['Medicaid'])
#feature_names.remove('Medicaid')

from scipy import stats
df_corr = pd.DataFrame() # Correlation matrix
df_p = pd.DataFrame()  # Matrix of p-values
for x in df3.columns:
    for y in df3.columns:
        corr = stats.pearsonr(df3[x], df3[y])
        df_corr.loc[x,y] = corr[0]
        df_p.loc[x,y] = corr[1]


df_corr.to_csv('/Users/bu/Desktop/df_corr.csv',header=1)
df_p.to_csv('/Users/bu/Desktop/df_p.csv',header=1)




##clustermap
#sns.set(font_scale=1.2)
df4 = df4.drop(columns=['PPD'])
corrmat = df4.corr() 
sns.set(font_scale=1.0)
sns.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1,figsize=(10, 10))


###########################################################################

y = df3['ppd']
X = df3.drop(columns=['ppd'])
X_train_sfslg, X_test_sfslg, y_trainlg, y_testlg = train_test_split(X, y, test_size=0.2, random_state=17)
print(X_test_sfslg.shape)
print(X_train_sfslg.shape)

#df.groupby(['unknowmarital']).size()
#df.groupby(['married']).size()
#df.groupby(['single']).size()

clfLogisticRegression.fit(X_train_sfslg, y_trainlg)
y_pred_c = clfLogisticRegression.predict(X_test_sfslg)
y_pred_proba_clg = clfLogisticRegression.predict_proba(X_test_sfslg)[:, 1]
confmat_test_c = confusion_matrix(y_true=y_testlg, y_pred=y_pred_c)

print('confmat_test:\n', confmat_test_c)
print('the acc is:', accuracy_score(y_testlg, y_pred_c))
print('the classification_report:', classification_report(y_testlg, y_pred_c))
print('the auc of logistics is:', roc_auc_score(y_testlg, y_pred_proba_clg))
print('the brier socre is',brier_score_loss(y_testlg, y_pred_proba_clg))

#(2919, 32)
#(11675, 32)
#confmat_test:
# [[2696  137]
# [  15   71]]
#the acc is: 0.947927372387804
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.95      0.97      2833
#        1.0       0.34      0.83      0.48        86
#
#avg / total       0.98      0.95      0.96      2919
#
#the auc of logistics is: 0.9196902781996239
#the brier socre is 0.11574609675920726

import pickle #pickle模块
#保存Model(注:save文件夹要预先建立，否则会报错)
with open('/Users/bu/Desktop/clflogistics.pickle', 'wb') as f:
    pickle.dump(clfLogisticRegression, f)
#读取Model
with open('/Users/bu/Desktop/clflogistics.pickle', 'rb') as f:
    clf2 = pickle.load(f)
    #测试读取后的Model
    print(clf2.predict(X[0:1]))

###########################################################################
#wcmmodel30
dfwcmmodel30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel30.csv', delimiter=',', header=0)
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
# [[2742   91]
# [  19   67]]
#the acc is: 0.9623158615964371
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2833
#        1.0       0.42      0.78      0.55        86
#
#avg / total       0.98      0.96      0.97      2919
#
#the auc of logistics is: 0.8927568770060499
#the brier socre is 0.0957489257112601



###########################################################################
#wcmmodel24
dfwcmmodel24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel24.csv', delimiter=',', header=0)
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
# [[2743   90]
# [  19   67]]
#the acc is: 0.9626584446728331
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2833
#        1.0       0.43      0.78      0.55        86
#
#avg / total       0.98      0.96      0.97      2919
#
#the auc of logistics is: 0.895525328561226
#the brier socre is 0.09516975838547727


###########################################################################
#wcmmodel18
dfwcmmodel18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel18.csv', delimiter=',', header=0)
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
# [[2744   89]
# [  19   67]]
#the acc is: 0.9630010277492291
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2833
#        1.0       0.43      0.78      0.55        86
#
#avg / total       0.98      0.96      0.97      2919
#
#the auc of logistics is: 0.8948521987538891
#the brier socre is 0.09492216026832227


###########################################################################
#wcmmodel12
dfwcmmodel12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel12.csv', delimiter=',', header=0)
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
# [[2743   90]
# [  19   67]]
#the acc is: 0.9626584446728331
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.97      0.98      2833
#        1.0       0.43      0.78      0.55        86
#
#avg / total       0.98      0.96      0.97      2919
#
#the auc of logistics is: 0.8963646885953752
#the brier socre is 0.0943933539493625



###########################################################################
#cdrn
columnslg = columnslg19 + columnslg25
columnslg = list(dict.fromkeys(columnslg))
columnslg.remove('C137.0')
columnslg.remove('C2544.0')
columnslg.remove('C2644.0')

dfcdrn = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew.csv', delimiter=',', header=0)
dfcdrn['ppd'].value_counts()
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
dfcdrn = pd.merge(dfcdrn,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn = pd.DataFrame(dfcdrn, columns=columnslg)
df2cdrn['abdominalpain'] = df2cdrn.apply(lambda x: abdominal(x['C11.0'], x['C2043.0']), axis = 1)
df2cdrn['primigravida'] = df2cdrn.apply(lambda x: primigravida(x['C1685.0'], x['C3830.0']), axis = 1)
df2cdrn['vomit'] = df2cdrn.apply(lambda x: vomit(x['C4879.0'], x['C3066.0']), axis = 1)
df2cdrn = df2cdrn.drop(['C11.0', 'C2043.0', 'C1685.0', 'C3830.0', 'C4879.0','C3066.0','C345.0','C2057.0'], axis=1)
df2cdrn['ppd'].value_counts()
#0.0    50424
#1.0     1889

df2cdrn[np.isnan(df2cdrn)] = '0'
ycdrn = df2cdrn['ppd']
Xcdrn = df2cdrn.drop(columns=['ppd'])

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
# [[42307  8117]
# [  232  1657]]
#the acc is: 0.8404029591115019
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.84      0.91     50424
#        1.0       0.17      0.88      0.28      1889
#
#avg / total       0.96      0.84      0.89     52313
#
#the auc of logistics is: 0.8865172883970399
#the brier socre is 0.18224850725161781

###########################################################################
#cdrn30
dfcdrn30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew30.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
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
# [[42595  7829]
# [  242  1647]]
#the acc is: 0.8457171257622388
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.84      0.91     50424
#        1.0       0.17      0.87      0.29      1889
#
#avg / total       0.96      0.85      0.89     52313
#
#the auc of logistics is: 0.8842533631375549
#the brier socre is 0.17437707509297104

###########################################################################
#cdrn24
dfcdrn24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew24.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
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
# [[42684  7740]
# [  244  1645]]
#the acc is: 0.8473801923040162
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.85      0.91     50424
#        1.0       0.18      0.87      0.29      1889
#
#avg / total       0.96      0.85      0.89     52313
#
#the auc of logistics is: 0.8841198316413394
#the brier socre is 0.17208904187422727

###########################################################################
#cdrn18
dfcdrn18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew18.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
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
# [[42741  7683]
# [  247  1642]]
#the acc is: 0.8484124405023608
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.85      0.92     50424
#        1.0       0.18      0.87      0.29      1889
#
#avg / total       0.96      0.85      0.89     52313
#
#the auc of logistics is: 0.8831104872292279
#the brier socre is 0.16979963589925617

###########################################################################
#cdrn12
dfcdrn12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew12.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
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
# [[42822  7602]
# [  251  1638]]
#the acc is: 0.8498843499703707
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.99      0.85      0.92     50424
#        1.0       0.18      0.87      0.29      1889
#
#avg / total       0.96      0.85      0.89     52313
#
#the auc of logistics is: 0.8814270707009114
#the brier socre is 0.16613541439451998
