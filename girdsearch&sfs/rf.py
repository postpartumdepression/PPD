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


#print(df)
y = df['ppd']
X = df.drop(columns=['person_id', 'ppd','condition_start_date'])
print('ok')
#print(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)


RandomForest = ensemble.RandomForestClassifier(class_weight='balanced',)
gridrf = GridSearchCV(RandomForest,
                      param_grid={'max_depth': range(2, 20, 1),
                                'min_samples_split':range(40, 150, 20),
                                'min_samples_leaf':range(1, 60, 20)},
                    scoring='roc_auc',
                    cv=5,
                    return_train_score=True)
gridrf.fit(X_train, y_train)
scores = pd.DataFrame(gridrf.cv_results_)
print(gridrf.best_estimator_)

clfRandomForest = ensemble.RandomForestClassifier(bootstrap=True, class_weight='balanced',
                       criterion='gini', max_depth=8, max_features='auto',
                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=21,
                       min_samples_split=140, min_weight_fraction_leaf=0.0,
                       n_estimators=10, n_jobs=None, oob_score=False,
                       random_state=None, verbose=0, warm_start=False)

clfRandomForest.fit(X_train, y_train)
y_pred_c = clfRandomForest.predict(X_test)
y_pred_proba_crf = clfRandomForest.predict_proba(X_test)[:, 1]
confmat_test_c = confusion_matrix(y_true=y_test, y_pred=y_pred_c)

print('confmat_test:\n', confmat_test_c)
print('the acc is:', accuracy_score(y_test, y_pred_c))
print('the classification_report:', classification_report(y_test, y_pred_c))
print('the auc of random forest is:', roc_auc_score(y_test, y_pred_proba_crf))

#confmat_test:
# [[2621  688]
# [  43   25]]
#the acc is: 0.7835356825584838
#the classification_report:               precision    recall  f1-score   support
#
#         0.0       0.98      0.79      0.88      3309
#         1.0       0.04      0.37      0.06        68
#
#    accuracy                           0.78      3377
#   macro avg       0.51      0.58      0.47      3377
#weighted avg       0.96      0.78      0.86      3377
#
#the auc of random forest is: 0.6638023749844453

feature_names = X.columns.values.tolist()
print(feature_names)

f_number = 100
sfs4 = SFS(clfRandomForest,
           k_features=f_number,
           forward=True,
           floating=False,
           scoring='roc_auc',
           cv=5)


result4 = sfs4.fit(X_train, y_train, custom_feature_names=feature_names)
#print(X)
result4.subsets_
result4.k_score_

selection_res = pd.DataFrame.from_dict(sfs4.get_metric_dict()).T
# print(selection_res)
selection_res.to_csv("/Users/shuojiawang/Documents/ppdmodel/result1907/selection_log_withouthistoryrf100.csv", sep='\t')

selected_feature_idx = result4.k_feature_idx_
#print(type(selected_feature_idx))
selected_feature = list(selected_feature_idx)
feature_name = []
for i in selected_feature:
    feature_name.append(feature_names[i])
print(feature_name)


fig = plot_sfs(sfs4.get_metric_dict(), kind='std_err')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.xlabel("Feature number")
plt.ylabel("AUC")
plt.grid()
#plt.savefig("Users/bu/Desktop/feature_selection.png", dpi=600)
plt.show()
#plt.clf()

