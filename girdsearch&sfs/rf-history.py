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

df = pd.read_csv('/Users/shuojiawang/Documents/ppdmodel/modelfilenew/wcmmodelwithhistory.csv', delimiter=',', header=0)

#print(df)
y = df['ppd']
X = df.drop(columns=['person_id', 'ppd','condition_start_date'])

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

feature_names = X.columns.values.tolist()
print(feature_names)
print('ok2')



clfRandomForest = ensemble.RandomForestClassifier(bootstrap=True, class_weight='balanced',
                       criterion='gini', max_depth=17, max_features='auto',
                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=1,
                       min_samples_split=120, min_weight_fraction_leaf=0.0,
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
# [[3181  128]
# [  38   30]]
#the acc is: 0.9508439443292863
#the classification_report:               precision    recall  f1-score   support
#
#         0.0       0.99      0.96      0.97      3309
#         1.0       0.19      0.44      0.27        68
#
#    accuracy                           0.95      3377
#   macro avg       0.59      0.70      0.62      3377
#weighted avg       0.97      0.95      0.96      3377
#
#the auc of random forest is: 0.8112633992853715

f_number = 50
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
selection_res.to_csv("/Users/shuojiawang/Documents/ppdmodel/result1907/selection_log_withistoryrf.csv", sep='\t')

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













from sklearn.model_selection import learning_curve
# Create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(ensemble.RandomForestClassifier(),
                                                        X,
                                                        y,
                                                        # Number of folds in cross-validation
                                                        cv=10,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1,
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 50))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()