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
df = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodelwithhistory.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visitexafter1ywcmnew.csv', sep=',')
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

condition_list = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/condition_listcombine.csv',delimiter=',', header=0)
medication_list = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/medication_listnew.csv',delimiter=',', header=0)
medication_list = medication_list.rename({'concept_name.1': 'concept_name_y'}, axis=1) #'a': 'X', 'b': 'Y'
name_list = pd.concat([medication_list,condition_list],axis = 0)

df3 = pd.DataFrame(df_2, columns=columnsxgboost)
feature_names = df3.columns.values.tolist()
columnsori = feature_names
columns = pd.DataFrame({'newvariable': columnsori})
columns2 = pd.merge(columns, name_list, on = ['newvariable'], how='left')
##verify 'Ethnic background'
#import pandas as pd
#folder="/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/orifile/wcm/"
#condition = pd.read_csv(folder+ "condition.csv", delimiter=',', header=0)
#eth = condition[(condition['concept_name']=='Ethnic background')]
#person = pd.read_csv(folder+ "person.csv", delimiter=',', header=0)
#eth2 = pd.merge(eth,person,on=['person_id'],how='left')
#'Ethnic background' is: not hispanic



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
  #'C4056.0',#Rheumatoid arthritis
  #'C4101.0',#Scar conditions and fibrosis of skin
 'C419.0',#Anxiety
 'M133.0',#TOPICAL PRODUCTS FOR JOINT AND MUSCULAR PAIN
 'M14.0',#ANTIDEPRESSANTS
 #'M32.0',ANTIPSYCHOTICS
 'M98.0']#OTHER PLAIN VITAMIN PREPARATIONS

df3 = pd.DataFrame(df_2, columns=columnsxgboost2)
y = df3['ppd']
X = df3.drop(columns=['ppd'])
X_train_sfsxgboost, X_test_sfsxgboost, y_trainxgboost, y_testxgboost = train_test_split(X, y, test_size=0.2, random_state=17)

clfxgboost.fit(X_train_sfsxgboost, y_trainxgboost)
y_pred_c = clfxgboost.predict(X_test_sfsxgboost)
y_pred_proba_cxg = clfxgboost.predict_proba(X_test_sfsxgboost)[:, 1]
confmat_test_c = confusion_matrix(y_true=y_testxgboost, y_pred=y_pred_c)

print('confmat_test:\n', confmat_test_c)
print('the acc is:', accuracy_score(y_testxgboost, y_pred_c))
print('the classification_report:\n', classification_report(y_testxgboost, y_pred_c))
print('the auc of xgboost is:', roc_auc_score(y_testxgboost, y_pred_proba_cxg))
print('the brier socre is',brier_score_loss(y_testxgboost, y_pred_proba_cxg))




#wcm
df3 = pd.DataFrame(df_2, columns=columnsxgboost)
df3.isnull().any()
#df3[np.isnan(df3)] = '0'
#df3.head(2)
y = df3['ppd']
X = df3.drop(columns=['ppd'])
X_train_sfsxgboost, X_test_sfsxgboost, y_trainxgboost, y_testxgboost = train_test_split(X, y, test_size=0.2, random_state=17)

df3_2 = df3
df3_2['a'] = 1
y3 = df3_2['ppd']
X3 = df3_2.drop(columns=['ppd'])
X_train_sfslg3, X_test_sfslg3, y_trainlg3, y_testlg3 = train_test_split(X3, y3, test_size=0.2, random_state=17)
dftrain = pd.concat([X_train_sfslg3, y_trainlg3],axis=1)
for i in columnsxgboost:
    print(dftrain.groupby(['ppd',i])['a'].count())
    
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
 #'C4056.0',#Rheumatoid arthritis
 #'C4101.0',#Scar conditions and fibrosis of skin
 'C419.0',#Anxiety
 'M133.0',#TOPICAL PRODUCTS FOR JOINT AND MUSCULAR PAIN
 'M14.0',#ANTIDEPRESSANTS
 #'M32.0',ANTIPSYCHOTICS
 'M98.0']#OTHER PLAIN VITAMIN PREPARATIONS

df3 = pd.DataFrame(df_2, columns=columnsxgboost2)
y = df3['ppd']
X = df3.drop(columns=['ppd'])
X_train_sfsxgboost, X_test_sfsxgboost, y_trainxgboost, y_testxgboost = train_test_split(X, y, test_size=0.2, random_state=17)

# from xgboost import plot_tree
# import matplotlib.pyplot as plt
# plot_tree(clfxgboost)
# plt.show()

clfxgboost.fit(X_train_sfsxgboost, y_trainxgboost)
y_pred_c = clfxgboost.predict(X_test_sfsxgboost)
y_pred_proba_cxg = clfxgboost.predict_proba(X_test_sfsxgboost)[:, 1]
confmat_test_c = confusion_matrix(y_true=y_testxgboost, y_pred=y_pred_c)

print('confmat_test:\n', confmat_test_c)
print('the acc is:', accuracy_score(y_testxgboost, y_pred_c))
print('the classification_report:\n', classification_report(y_testxgboost, y_pred_c))
print('the auc of xgboost is:', roc_auc_score(y_testxgboost, y_pred_proba_cxg))
print('the brier socre is',brier_score_loss(y_testxgboost, y_pred_proba_cxg))

#fit withouthistory
# confmat_test:
#  [[2699  134]
#  [  15   71]]
# the acc is: 0.9489551216169921
# the classification_report:
#                precision    recall  f1-score   support
#          0.0       0.99      0.95      0.97      2833
#          1.0       0.35      0.83      0.49        86
#    micro avg       0.95      0.95      0.95      2919
#    macro avg       0.67      0.89      0.73      2919
# weighted avg       0.98      0.95      0.96      2919
# the auc of xgboost is: 0.917500554100756
# the brier socre is 0.0718502600704142
import pickle
with open('/Users/bu/Desktop/clfxgboost.pickle', 'wb') as f:
    pickle.dump(clfxgboost, f)
# with open('/Users/bu/Desktop/clfxgboost.pickle', 'rb') as f:
#        clf2 = pickle.load(f)
#        #测试读取后的Model
#        print(clf2.predict(X[0:1])
###########################################################################
#wcmmodel30
dfwcmmodel30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel30.csv', delimiter=',', header=0)
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
#  [[2734   99]
#  [  17   69]]
# the acc is: 0.960260363138061
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.99      0.97      0.98      2833
#          1.0       0.41      0.80      0.54        86
#    micro avg       0.96      0.96      0.96      2919
#    macro avg       0.70      0.88      0.76      2919
# weighted avg       0.98      0.96      0.97      2919
# the auc of xgboost is: 0.9060696607261594
# the brier socre is 0.06409347617084793


###########################################################################
#wcmmodel24
dfwcmmodel24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel24.csv', delimiter=',', header=0)
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
print('the auc of xgboost is:', roc_auc_score(y_test24, y_pred_proba_cxgboost))
print('the brier socre is',brier_score_loss(y_test24, y_pred_proba_cxgboost))

#withouthistory
# confmat_test:
#  [[2711  122]
#  [  17   69]]
# the acc is: 0.9523809523809523
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.99      0.96      0.98      2833
#          1.0       0.36      0.80      0.50        86
#    micro avg       0.95      0.95      0.95      2919
#    macro avg       0.68      0.88      0.74      2919
# weighted avg       0.98      0.95      0.96      2919
# the auc of mlp is: 0.8891018642412103
# the brier socre is 0.0653802348839138

###########################################################################
#wcmmodel18
dfwcmmodel18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel18.csv', delimiter=',', header=0)
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
#  [[2747   86]
#  [  17   69]]
# the acc is: 0.9647139431312093
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.99      0.97      0.98      2833
#          1.0       0.45      0.80      0.57        86
#    micro avg       0.96      0.96      0.96      2919
#    macro avg       0.72      0.89      0.78      2919
# weighted avg       0.98      0.96      0.97      2919
# the auc of xbgoost is: 0.8885272412349469
# the brier socre is 0.061089757407609045

###########################################################################
#wcmmodel12
dfwcmmodel12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel12.csv', delimiter=',', header=0)
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
#  [[2749   84]
#  [  17   69]]
# the acc is: 0.9653991092840014
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.99      0.97      0.98      2833
#          1.0       0.45      0.80      0.58        86
#    micro avg       0.97      0.97      0.97      2919
#    macro avg       0.72      0.89      0.78      2919
# weighted avg       0.98      0.97      0.97      2919
# the auc of xbgoost is: 0.8895677193212882
# the brier socre is 0.047316528576188496


###########################################################################
#cdrn
dfcdrn = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew.csv', delimiter=',', header=0)
dfcdrn['ppd'].value_counts()
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
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
#  [[42717  7707]
#  [  221  1668]]
# the acc is: 0.8484506719171143
# the classification_report:
#                precision    recall  f1-score   support
#          0.0       0.99      0.85      0.92     50424
#          1.0       0.18      0.88      0.30      1889
#    micro avg       0.85      0.85      0.85     52313
#    macro avg       0.59      0.87      0.61     52313
# weighted avg       0.97      0.85      0.89     52313
# the auc of logistics is: 0.8787454697558036
# the brier socre is 0.15091797502951845

###########################################################################
#cdrn30
dfcdrn30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew30.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
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

#without history
# confmat_test:
#  [[42752  7672]
#  [  229  1660]]
# the acc is: 0.8489667960162866
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.99      0.85      0.92     50424
#          1.0       0.18      0.88      0.30      1889
#    micro avg       0.85      0.85      0.85     52313
#    macro avg       0.59      0.86      0.61     52313
# weighted avg       0.97      0.85      0.89     52313
# the auc of logistics is: 0.8730634101065423
# the brier socre is 0.15141424893711133

###########################################################################
#cdrn24
dfcdrn24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew24.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
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

#fit history
# confmat_test:
#  [[42771  7653]
#  [  230  1659]]
# the acc is: 0.8493108787490681
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.99      0.85      0.92     50424
#          1.0       0.18      0.88      0.30      1889
#    micro avg       0.85      0.85      0.85     52313
#    macro avg       0.59      0.86      0.61     52313
# weighted avg       0.97      0.85      0.89     52313
# the auc of logistics is: 0.871583293417715
# the brier socre is 0.15114698588751488
###########################################################################
#cdrn18
dfcdrn18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew18.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
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
#  [[42786  7638]
#  [  235  1654]]
# the acc is: 0.8495020358228357
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.99      0.85      0.92     50424
#          1.0       0.18      0.88      0.30      1889
#    micro avg       0.85      0.85      0.85     52313
#    macro avg       0.59      0.86      0.61     52313
# weighted avg       0.97      0.85      0.89     52313
# the auc of logistics is: 0.8702715582763407
# the brier socre is 0.1509902799036681

###########################################################################
#cdrn12
dfcdrn12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew12.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew.csv', delimiter=',', header=0)
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
#  [[42862  7562]
#  [  245  1644]]
# the acc is: 0.8507636725097012
# the classification_report:               precision    recall  f1-score   support
#          0.0       0.99      0.85      0.92     50424
#          1.0       0.18      0.87      0.30      1889
#    micro avg       0.85      0.85      0.85     52313
#    macro avg       0.59      0.86      0.61     52313
# weighted avg       0.96      0.85      0.89     52313
# the auc of logistics is: 0.8708206394948184
# the brier socre is 0.15020906565275843

