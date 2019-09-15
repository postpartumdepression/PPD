import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, accuracy_score, auc, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import ensemble
from sklearn import tree
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import brier_score_loss

df = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodelwithhistory_2.csv', delimiter=',', header=0)
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visitexafter1ywcmnew_2.csv', sep=',')
visitexafter1y = pd.DataFrame(visitexafter1y,columns=['person_id','condition_start_date'])
df_2 = pd.merge(df,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
y = df_2['ppd']
X = df_2.drop(columns=['person_id', 'condition_start_date','ppd'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

columnsmlp = X_train.columns.values.tolist()
columnsmlp = columnsmlp + ['ppd']

mlp = MLPClassifier()
parameter_space = {
    'hidden_layer_sizes': [(50,50),(50, 50, 50), (50, 100, 50)],
    #'hidden_layer_sizes': [(64, 512), (64, 512), (64, 512)], (512, 64, 32)
    'activation': ['tanh', 'relu'],
    #'activation': ['tanh', 'relu', 'sigmoid','linear'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train, y_train)
# Best paramete set
print('Best parameters found:\n', clf.best_params_)
#{'alpha': 0.05, 'activation': 'relu', 'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': (50, 100, 50)}


clf = MLPClassifier(activation= 'tanh', alpha= 0.0001, 
                    hidden_layer_sizes= (64,512), random_state=1,
                    learning_rate= 'adaptive', solver =  'sgd')

clf.fit(X_train, y_train)
y_pred_c = clf.predict(X_test)
y_pred_proba_c = clf.predict_proba(X_test)[:, 1]
confmat_test_c = confusion_matrix(y_true=y_test, y_pred=y_pred_c)

print('confmat_test:\n', confmat_test_c)
print('the acc is:', accuracy_score(y_test, y_pred_c))
print('the classification_report:', classification_report(y_test, y_pred_c))
print('the auc of mlp is:', roc_auc_score(y_test, y_pred_proba_c))
print('the brier socre is',brier_score_loss(y_test, y_pred_proba_c))

#confmat_test:
# [[2824   39]
# [  63  114]]
#the acc is: 0.9664473684210526
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.99      0.98      2863
#        1.0       0.75      0.64      0.69       177
#
#avg / total       0.96      0.97      0.97      3040
#
#the auc of mlp is: 0.9333380693871348
#the brier socre is 0.025957603020181307
import pickle 
with open('/Users/bu/Desktop/clfmlp_2.pickle', 'wb') as f:
    pickle.dump(clf, f)
    
#wcmmodel30
dfwcmmodel30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel30_2.csv', delimiter=',', header=0)
dfwcmmodel30 = pd.merge(dfwcmmodel30,visitexafter1y,on = ['person_id','condition_start_date'], how='right')

df2wcmmodel30 = pd.DataFrame(dfwcmmodel30, columns=columnsmlp)
df2wcmmodel30[np.isnan(df2wcmmodel30)] = '0'

y30 = df2wcmmodel30['ppd']
X30 = df2wcmmodel30.drop(columns=['ppd'])

X_train30, X_test30, y_train30, y_test30 = train_test_split(X30, y30, test_size=0.2, random_state=17)

clf.fit(X_train, y_train)
y_pred_crf = clf.predict(X_test30)
y_pred_proba_crf = clf.predict_proba(X_test30)[:, 1]
confmat_test_crf = confusion_matrix(y_true=y_test30, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(y_test30, y_pred_crf))
print('the classification_report:', classification_report(y_test30, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(y_test30, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(y_test30, y_pred_proba_crf))

#confmat_test:
# [[2819   44]
# [  64  113]]
#the acc is: 0.9644736842105263
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.98      0.98      2863
#        1.0       0.72      0.64      0.68       177
#
#avg / total       0.96      0.96      0.96      3040
#
#the auc of random forest is: 0.9173755947200892
#the brier socre is 0.027518566221844516

#wcmmodel24
dfwcmmodel24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel24_2.csv', delimiter=',', header=0)
dfwcmmodel24 = pd.merge(dfwcmmodel24,visitexafter1y,on = ['person_id','condition_start_date'], how='right')

df2wcmmodel24 = pd.DataFrame(dfwcmmodel24, columns=columnsmlp)
df2wcmmodel24[np.isnan(df2wcmmodel24)] = '0'

y24 = df2wcmmodel24['ppd']
X24 = df2wcmmodel24.drop(columns=['ppd'])

X_train24, X_test24, y_train24, y_test24 = train_test_split(X24, y24, test_size=0.2, random_state=17)

clf.fit(X_train, y_train)
y_pred_crf = clf.predict(X_test24)
y_pred_proba_crf = clf.predict_proba(X_test24)[:, 1]
confmat_test_crf = confusion_matrix(y_true=y_test24, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(y_test24, y_pred_crf))
print('the classification_report:', classification_report(y_test24, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(y_test24, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(y_test24, y_pred_proba_crf))
#
#confmat_test:
# [[2820   43]
# [  65  112]]
#the acc is: 0.9644736842105263
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.98      0.98      2863
#        1.0       0.72      0.63      0.67       177
#
#avg / total       0.96      0.96      0.96      3040
#
#the auc of random forest is: 0.9195877265165733
#the brier socre is 0.027692912969775964

#wcmmodel18
dfwcmmodel18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel18_2.csv', delimiter=',', header=0)
dfwcmmodel18 = pd.merge(dfwcmmodel18,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel18 = pd.DataFrame(dfwcmmodel18, columns=columnsmlp)
df2wcmmodel18[np.isnan(df2wcmmodel18)] = '0'

y18 = df2wcmmodel18['ppd']
X18 = df2wcmmodel18.drop(columns=['ppd'])
X_train18, X_test18, y_train18, y_test18 = train_test_split(X18, y18, test_size=0.2, random_state=17)

clf.fit(X_train, y_train)
y_pred_crf = clf.predict(X_test18)
y_pred_proba_crf = clf.predict_proba(X_test18)[:, 1]
confmat_test_crf = confusion_matrix(y_true=y_test18, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(y_test18, y_pred_crf))
print('the classification_report:', classification_report(y_test18, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(y_test18, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(y_test18, y_pred_proba_crf))

#confmat_test:
# [[2818   45]
# [  65  112]]
#the acc is: 0.9638157894736842
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.98      0.98      2863
#        1.0       0.71      0.63      0.67       177
#
#avg / total       0.96      0.96      0.96      3040
#
#the auc of random forest is: 0.9189740128781198
#the brier socre is 0.027801559847288985

#wcmmodel12
dfwcmmodel12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/wcm/wcmmodel12_2.csv', delimiter=',', header=0)
dfwcmmodel12 = pd.merge(dfwcmmodel12,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2wcmmodel12 = pd.DataFrame(dfwcmmodel12, columns=columnsmlp)
df2wcmmodel12[np.isnan(df2wcmmodel12)] = '0'

y12 = df2wcmmodel12['ppd']
X12 = df2wcmmodel12.drop(columns=['ppd'])

X_train12, X_test12, y_train12, y_test12 = train_test_split(X12, y12, test_size=0.2, random_state=17)

clf.fit(X_train, y_train)
y_pred_crf = clf.predict(X_test12)
y_pred_proba_crf = clf.predict_proba(X_test12)[:, 1]
confmat_test_crf = confusion_matrix(y_true=y_test12, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(y_test12, y_pred_crf))
print('the classification_report:', classification_report(y_test12, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(y_test12, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(y_test12, y_pred_proba_crf))

#confmat_test:
# [[2817   46]
# [  66  111]]
#the acc is: 0.9631578947368421
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.98      0.98      0.98      2863
#        1.0       0.71      0.63      0.66       177
#
#avg / total       0.96      0.96      0.96      3040
#
#the auc of random forest is: 0.9205102703300043
#the brier socre is 0.02813249102030993

#cdrn
dfcdrn = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew_2.csv', delimiter=',', header=0)
dfcdrn['ppd'].value_counts()
visitexafter1y = pd.read_csv('/Users/bu/Desktop/visit2cdrnnew_2.csv', delimiter=',', header=0)
dfcdrn = pd.merge(dfcdrn,visitexafter1y,on = ['person_id','condition_start_date'], how='inner')
df2cdrn = pd.DataFrame(dfcdrn, columns=columnsmlp)
df2cdrn[np.isnan(df2cdrn)] = '0'

ycdrn = df2cdrn['ppd']
Xcdrn = df2cdrn.drop(columns=['ppd'])

clf.fit(X_train, y_train)
y_pred_crf = clf.predict(Xcdrn)
y_pred_proba_crf = clf.predict_proba(Xcdrn)[:, 1]
confmat_test_crf = confusion_matrix(y_true=ycdrn, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(ycdrn, y_pred_crf))
print('the classification_report:', classification_report(ycdrn, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(ycdrn, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(ycdrn, y_pred_proba_crf))

#confmat_test:
# [[44622  5837]
# [ 1210  2303]]
#the acc is: 0.8694322982287112
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.97      0.88      0.93     50459
#        1.0       0.28      0.66      0.40      3513
#
#avg / total       0.93      0.87      0.89     53972
#
#the auc of random forest is: 0.887111155910969
#the brier socre is 0.10524862976565269

#cdrn30
dfcdrn30 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew30_2.csv', delimiter=',', header=0)
dfcdrn30 = pd.merge(dfcdrn30,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn30 = pd.DataFrame(dfcdrn30, columns=columnsmlp)
df2cdrn30[np.isnan(df2cdrn30)] = '0'

ycdrn = df2cdrn30['ppd']
Xcdrn = df2cdrn30.drop(columns=['ppd'])

clf.fit(X_train, y_train)
y_pred_crf = clf.predict(Xcdrn)
y_pred_proba_crf = clf.predict_proba(Xcdrn)[:, 1]
confmat_test_crf = confusion_matrix(y_true=ycdrn, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(ycdrn, y_pred_crf))
print('the classification_report:', classification_report(ycdrn, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(ycdrn, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(ycdrn, y_pred_proba_crf))

#confmat_test:
# [[44233  6226]
# [ 1243  2270]]
#the acc is: 0.8616134291855035
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.97      0.88      0.92     50459
#        1.0       0.27      0.65      0.38      3513
#
#avg / total       0.93      0.86      0.89     53972
#
#the auc of random forest is: 0.8265935083708387
#the brier socre is 0.11033129126328747


#cdrn24
dfcdrn24 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew24_2.csv', delimiter=',', header=0)
dfcdrn24 = pd.merge(dfcdrn24,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn24 = pd.DataFrame(dfcdrn24, columns=columnsmlp)
df2cdrn24[np.isnan(df2cdrn24)] = '0'

ycdrn = df2cdrn24['ppd']
Xcdrn = df2cdrn24.drop(columns=['ppd'])

clf.fit(X_train, y_train)
y_pred_crf = clf.predict(Xcdrn)
y_pred_proba_crf = clf.predict_proba(Xcdrn)[:, 1]
confmat_test_crf = confusion_matrix(y_true=ycdrn, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(ycdrn, y_pred_crf))
print('the classification_report:', classification_report(ycdrn, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(ycdrn, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(ycdrn, y_pred_proba_crf))

#confmat_test:
# [[44218  6241]
# [ 1249  2264]]
#the acc is: 0.8612243385459127
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.97      0.88      0.92     50459
#        1.0       0.27      0.64      0.38      3513
#
#avg / total       0.93      0.86      0.89     53972
#
#the auc of random forest is: 0.824150357221419
#the brier socre is 0.10999314001916023

#cdrn18
dfcdrn18 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew18_2.csv', delimiter=',', header=0)
dfcdrn18 = pd.merge(dfcdrn18,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn18 = pd.DataFrame(dfcdrn18, columns=columnsmlp)
df2cdrn18[np.isnan(df2cdrn18)] = '0'

ycdrn = df2cdrn18['ppd']
Xcdrn = df2cdrn18.drop(columns=['ppd'])

clf.fit(X_train, y_train)
y_pred_crf = clf.predict(Xcdrn)
y_pred_proba_crf = clf.predict_proba(Xcdrn)[:, 1]
confmat_test_crf = confusion_matrix(y_true=ycdrn, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(ycdrn, y_pred_crf))
print('the classification_report:', classification_report(ycdrn, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(ycdrn, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(ycdrn, y_pred_proba_crf))

#confmat_test:
# [[44100  6359]
# [ 1235  2278]]
#the acc is: 0.859297413473653
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.97      0.87      0.92     50459
#        1.0       0.26      0.65      0.37      3513
#
#avg / total       0.93      0.86      0.89     53972
#
#the auc of random forest is: 0.8165154922502573
#the brier socre is 0.11100837415586276


#cdrn12
dfcdrn12 = pd.read_csv('/Users/bu/Documents/WCMC/PPD/ppdmodel1907/data/modelfile/cdrn/cdrnnew12_2.csv', delimiter=',', header=0)
dfcdrn12 = pd.merge(dfcdrn12,visitexafter1y,on = ['person_id','condition_start_date'], how='right')
df2cdrn12 = pd.DataFrame(dfcdrn12, columns=columnsmlp)
df2cdrn12[np.isnan(df2cdrn12)] = '0'

ycdrn = df2cdrn12['ppd']
Xcdrn = df2cdrn12.drop(columns=['ppd'])

clf.fit(X_train, y_train)
y_pred_crf = clf.predict(Xcdrn)
y_pred_proba_crf = clf.predict_proba(Xcdrn)[:, 1]
confmat_test_crf = confusion_matrix(y_true=ycdrn, y_pred=y_pred_crf)

print('confmat_test:\n', confmat_test_crf)
print('the acc is:', accuracy_score(ycdrn, y_pred_crf))
print('the classification_report:', classification_report(ycdrn, y_pred_crf))
print('the auc of random forest is:', roc_auc_score(ycdrn, y_pred_proba_crf))
print('the brier socre is',brier_score_loss(ycdrn, y_pred_proba_crf))

#confmat_test:
# [[44044  6415]
# [ 1225  2288]]
#the acc is: 0.858445119691692
#the classification_report:              precision    recall  f1-score   support
#
#        0.0       0.97      0.87      0.92     50459
#        1.0       0.26      0.65      0.37      3513
#
#avg / total       0.93      0.86      0.88     53972
#
#the auc of random forest is: 0.8120949258818564
#the brier socre is 0.11066710432598452
