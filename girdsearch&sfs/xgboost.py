
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
from sklearn.pipeline import Pipeline

df = pd.read_csv('/Users/shuojiawang/Documents/ppdmodel/modelfilenew/wcmmodelnohistory.csv', delimiter=',', header=0)

y3 = df['ppd']
X3 = df.drop(columns=['person_id', 'ppd','condition_start_date'])

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=17)

print(X_test.shape)
print(X_train.shape)

print(y_train.value_counts())
print(y_test.value_counts())

scale_pos_weight = np.sum(y3 == 0)/float(np.sum(y3 == 1))
XGboost = XGBClassifier(class_weight="balanced",scale_pos_weight=39.58894230769231)
param_test1 = {
    'max_depth':[5,7,9],
    'min_child_weight':[1,3]
}
gsearch1 = GridSearchCV(estimator = XGboost,
                       param_grid = param_test1, scoring='roc_auc', cv=5)
gsearch1.fit(X_train, y_train)
print(gsearch1.best_estimator_)



clfXGboost = XGBClassifier(base_score=0.5, booster='gbtree', class_weight='balanced',
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=39.58894230769231,
              seed=None, silent=None, subsample=1, verbosity=1)

#clfXGboost = XGBClassifier()

clfXGboost.fit(X_train, y_train)
y_pred_c = clfXGboost.predict(X_test)
y_pred_proba_cdt =clfXGboost.predict_proba(X_test)[:, 1]
confmat_test_c = confusion_matrix(y_true=y_test, y_pred=y_pred_c)
print('confmat_test:\n', confmat_test_c)
print('the acc is:', accuracy_score(y_test, y_pred_c))
print('the classification_report:', classification_report(y_test, y_pred_c))
print('the auc of XGboost is:', roc_auc_score(y_test, y_pred_proba_cdt))

#confmat_test:
# [[2956  353]
# [  51   17]]
#the acc is: 0.8803671898134439
#the classification_report:               precision    recall  f1-score   support
#
#         0.0       0.98      0.89      0.94      3309
#         1.0       0.05      0.25      0.08        68
#
#    accuracy                           0.88      3377
#   macro avg       0.51      0.57      0.51      3377
#weighted avg       0.96      0.88      0.92      3377
#
#the auc of XGboost is: 0.6486987360674098



feature_names = X3.columns.values.tolist()
#print(feature_names)
#print('ok2')

f_number = 100
sfs = SFS(clfXGboost,
           k_features=f_number,
           forward=True,
           floating=False,
           scoring='roc_auc',
           cv=5)

print('ok3')
result2 = sfs.fit(X_train, y_train, custom_feature_names=feature_names)
#print(X)
result2.subsets_
result2.k_score_

selection_res = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
# print(selection_res)
selection_res.to_csv("/Users/shuojiawang/Documents/ppdmodel/result1907/selection_log_withouthistoryxgboost.csv", sep='\t')

selected_feature_idx = result2.k_feature_idx_
#print(type(selected_feature_idx))
selected_feature = list(selected_feature_idx)
feature_name = []
for i in selected_feature:
    feature_name.append(feature_names[i])
print(feature_name)


fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.xlabel("Feature number")
plt.ylabel("AUC")
plt.grid()
plt.show()
#plt.clf()
