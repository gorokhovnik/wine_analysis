import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy.sparse import hstack
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from FE import FE
from RS import RS
from modeling.split import split

wine = pd.read_csv('../../wine.csv')

wine = FE(wine, 500)
wine = RS(wine)

tfidf = TfidfVectorizer(min_df=20,
                        ngram_range=(1, 1),
                        norm='l2',
                        max_features=5000)


X, X_d, X_f, y, \
X_train, X_train_d, X_train_f, y_train, \
X_test, X_test_d, X_test_f, y_test = split(wine=wine,
                                           description='description',
                                           features=['country', 'continent', 'price', 'category', 'year'],
                                           y='color',
                                           tfidf=tfidf)

AUCs = []


# # pca = PCA(n_components=1000,
# #           random_state=228)
# # pca.fit(X_train_d.toarray())
# # X_train_d = pca.transform(X_train_d.toarray())
# # X_test_d = pca.transform(X_test_d.toarray())
#
#
# model_logit = LogisticRegression(penalty='l1',
#                                  solver='liblinear',
#                                  C=1,
#                                  tol=1e-3,
#                                  max_iter=100,
#                                  random_state=228)
# model_logit.fit(X_train_d, y_train)
# p_train = model_logit.predict_proba(X_train_d)[:, 1]
# p_test = model_logit.predict_proba(X_test_d)[:, 1]
# AUCs += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]
#
# print(AUCs)
#
#
# model_forest = RandomForestClassifier(n_estimators=100,
#                                       oob_score=True,
#                                       max_depth=100,
#                                       min_samples_leaf=1,
#                                       verbose=1,
#                                       random_state=228)
# model_forest.fit(X_train_d, y_train)
# p_train = model_forest.predict_proba(X_train_d)[:, 1]
# p_test = model_forest.predict_proba(X_test_d)[:, 1]
#
# AUCs += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]
#
# print(AUCs)

print('lgb on description')
params = {
    'max_bin': 7,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 0.85,
    'bagging_fraction': 0.8,
    'learning_rate': 0.05,
    'max_depth': 25,
    'metric': 'auc',
    'min_data_in_leaf': 1,
    'min_sum_hessian_in_leaf': 0,
    'num_leaves': 101,
    'num_threads': 8,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'tree_learner': 'serial',
    'objective': 'binary'
}

to_train = lgb.Dataset(X_train_d, y_train)
to_val = lgb.Dataset(X_test_d, y_test)
model_lgb = lgb.train(params,
                      to_train,
                      valid_sets=[to_train, to_val],
                      num_boost_round=10000,
                      verbose_eval=100,
                      early_stopping_rounds=100)
p_train_d = model_lgb.predict(X_train_d)
p_test_d = model_lgb.predict(X_test_d)

AUCs += [[roc_auc_score(y_train, p_train_d), roc_auc_score(y_test, p_test_d)]]

print(AUCs)
model_lgb.save_model('description')


# pca = PCA(n_components=1000,
#           random_state=228)
# pca.fit(X_train_f.toarray())
# X_train_d = pca.transform(X_train_f.toarray())
# X_test_d = pca.transform(X_test_f.toarray())


# model_logit = LogisticRegression(penalty='l1',
#                                  solver='liblinear',
#                                  C=0.1,
#                                  tol=1e-4,
#                                  max_iter=100,
#                                  random_state=228)
# model_logit.fit(X_train_f, y_train)
# p_train = model_logit.predict_proba(X_train_f)[:, 1]
# p_test = model_logit.predict_proba(X_test_f)[:, 1]
#
# AUCs += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]
#
# print(AUCs)
#
#
# model_forest = RandomForestClassifier(n_estimators=200,
#                                       oob_score=True,
#                                       max_depth=13,
#                                       min_samples_leaf=2,
#                                       random_state=228)
# model_forest.fit(X_train_f, y_train)
# p_train = model_forest.predict_proba(X_train_f)[:, 1]
# p_test = model_forest.predict_proba(X_test_f)[:, 1]
#
# AUCs += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]
#
# print(AUCs)


print('lgb on feat')
params = {
    'max_bin': 255,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 1,
    'bagging_fraction': 0.8,
    'learning_rate': 0.05,
    'max_depth': 25,
    'metric': 'auc',
    'min_data_in_leaf': 1,
    'min_sum_hessian_in_leaf': 0,
    'num_leaves': 101,
    'num_threads': 8,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'tree_learner': 'serial',
    'objective': 'binary'
}

to_train = lgb.Dataset(X_train_f, y_train)
to_val = lgb.Dataset(X_test_f, y_test)
model_lgb = lgb.train(params,
                      to_train,
                      valid_sets=[to_train, to_val],
                      num_boost_round=1000,
                      verbose_eval=10,
                      early_stopping_rounds=20)
p_train_f = model_lgb.predict(X_train_f)
p_test_f = model_lgb.predict(X_test_f)

AUCs += [[roc_auc_score(y_train, p_train_f), roc_auc_score(y_test, p_test_f)]]

print(AUCs)
model_lgb.save_model('feat')

print('lgb on full')
params = {
    'max_bin': 127,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 0.85,
    'bagging_fraction': 0.8,
    'learning_rate': 0.05,
    'max_depth': 25,
    'metric': 'auc',
    'min_data_in_leaf': 1,
    'min_sum_hessian_in_leaf': 0,
    'num_leaves': 101,
    'num_threads': 8,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'tree_learner': 'serial',
    'objective': 'binary'
}

to_train = lgb.Dataset(X_train, y_train)
to_val = lgb.Dataset(X_test, y_test)
model_lgb = lgb.train(params,
                      to_train,
                      valid_sets=[to_train, to_val],
                      num_boost_round=10000,
                      verbose_eval=100,
                      early_stopping_rounds=100)
p_train = model_lgb.predict(X_train)
p_test = model_lgb.predict(X_test)

AUCs += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]

print(AUCs)
model_lgb.save_model('full')