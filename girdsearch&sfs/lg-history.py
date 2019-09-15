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


df = pd.read_csv('/Users/shuojiawang/Documents/ppdmodel/modelfilenew/wcmmodelwithhistory.csv', delimiter=',', header=0)
df['ppd'].value_counts()
#print(df)
y = df['ppd']
X = df.drop(columns=['person_id', 'ppd','condition_start_date'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
print(df.head(1))
print(X_test.shape)
print(X_train.shape)

print(y_train.value_counts())
print(y_test.value_counts())


feature_names = X.columns.values.tolist()#[2:]
print(feature_names)
print('ok2')

#grid search
logistics = LogisticRegression(class_weight="balanced")
gridlg = {#"C":np.logspace(-3,3,7),
      "C":[0.001,.009,0.01,0.1,.09,1,5,10,25],
      "penalty":["l1","l2"]}# l1 lasso l2 ridg
gridlg_cv=GridSearchCV(logistics,gridlg,scoring = 'roc_auc',cv=5)
gridlg_cv.fit(X_train,y_train)
print("tuned hpyerparameters :(best parameters) ",gridlg_cv.best_estimator_)


clfLogisticRegression = LogisticRegression(C=0.09, class_weight='balanced', dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='warn', n_jobs=None, penalty='l1',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)

clfLogisticRegression.fit(X_train, y_train)
y_pred_c = clfLogisticRegression.predict(X_test)
y_pred_proba_clg = clfLogisticRegression.predict_proba(X_test)[:, 1]
confmat_test_c = confusion_matrix(y_true=y_test, y_pred=y_pred_c)

print('confmat_test:\n', confmat_test_c)
print('the acc is:', accuracy_score(y_test, y_pred_c))
print('the classification_report:', classification_report(y_test, y_pred_c))
print('the auc of logistics is:', roc_auc_score(y_test, y_pred_proba_clg))
print('the brier socre is',brier_score_loss(y_test, y_pred_proba_clg))

#confmat_test:
# [[3125  184]
# [  17   51]]
#the acc is: 0.9404797157240155
#the classification_report:               precision    recall  f1-score   support
#
#         0.0       0.99      0.94      0.97      3309
#         1.0       0.22      0.75      0.34        68
#
#    accuracy                           0.94      3377
#   macro avg       0.61      0.85      0.65      3377
#weighted avg       0.98      0.94      0.96      3377
#
#the auc of logistics is: 0.8805752582084511
#the brier socre is 0.06334206090740635


f_number = 50
sfs = SFS(clfLogisticRegression,
           k_features=f_number,
           forward=True,
           floating=False,
           scoring='roc_auc',
           cv=5)
print('ok3')
result = sfs.fit(X_train, y_train, custom_feature_names=feature_names)
#print(X)
result.subsets_
result.k_score_

selection_res = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
# print(selection_res)
selection_res.to_csv("/Users/shuojiawang/Documents/ppdmodel/result1907/selection_log_withhistorylg.csv", sep='\t')

selected_feature_idx = result.k_feature_idx_
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
#plt.savefig("Users/bu/Desktop/feature_selection.png", dpi=600)
plt.show()
#plt.clf()














##### learning_curve#####

from sklearn.model_selection import learning_curve
# Create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(clfLogisticRegression,
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