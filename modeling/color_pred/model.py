import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score

from FE import FE
from RS import RS
from modeling.split import split

wine = pd.read_csv('../../wine.csv')

wine = FE(wine, 500)
wine = RS(wine)
wine['color'] = wine['color'].map({'red': 0, 'white': 1})

wine_summary = pd.read_csv('../../wine_summary.csv')
wine_summary.set_index('id', inplace=True)


tfidf = TfidfVectorizer(min_df=20,
                        ngram_range=(1, 1),
                        norm='l2',
                        max_features=5000)

dictv = DictVectorizer()

X_d, X_f, y, \
X_train_d, X_train_f, y_train, \
X_test_d, X_test_f, y_test = split(wine=wine,
                                   description='description',
                                   features=['country', 'continent', 'price', 'category', 'year'],
                                   y='color',
                                   tfidf=tfidf,
                                   dictv=dictv)

accuracy = []
roc_auc = []


# pca = PCA(n_components=1000,
#           random_state=228)
# pca.fit(X_train_d.toarray())
# X_train_d = pca.transform(X_train_d.toarray())
# X_test_d = pca.transform(X_test_d.toarray())

# model_perceptron = Perceptron(max_iter=200,
#                               penalty=None,
#                               early_stopping=True,
#                               validation_fraction=0.05,
#                               random_state=228)
# model_perceptron.fit(X_train_d, y_train)
# p_train = model_perceptron.predict(X_train_d)
# p_test = model_perceptron.predict(X_test_d)
#
# accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
# roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]
#
# print(accuracy)
# print(roc_auc)


# model_logit = LogisticRegression(penalty='l1',
#                                  solver='liblinear',
#                                  C=1,
#                                  tol=1e-3,
#                                  max_iter=100,
#                                  random_state=228)
# model_logit.fit(X_train_d, y_train)
# p_train = model_logit.predict(X_train_d)
# p_test = model_logit.predict(X_test_d)
#
# accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
# roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]
#
# print(accuracy)
# print(roc_auc)


# model_forest = RandomForestClassifier(n_estimators=100,
#                                       oob_score=True,
#                                       max_depth=100,
#                                       min_samples_leaf=1,
#                                       verbose=1,
#                                       random_state=228)
# model_forest.fit(X_train_d, y_train)
# p_train = model_forest.predict(X_train_d)
# p_test = model_forest.predict(X_test_d)
#
# accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
# roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]
#
# print(accuracy)
# print(roc_auc)

print('lgb on text')

# params = {
#     'max_bin': 7,
#     'boost_from_average': 'false',
#     'boost': 'gbdt',
#     'feature_fraction': 0.85,
#     'bagging_fraction': 0.8,
#     'learning_rate': 0.05,
#     'max_depth': 25,
#     'metric': 'auc',
#     'min_data_in_leaf': 1,
#     'min_sum_hessian_in_leaf': 0,
#     'num_leaves': 101,
#     'num_threads': 8,
#     'lambda_l1': 0,
#     'lambda_l2': 0,
#     'tree_learner': 'serial',
#     'objective': 'binary'
#
# }
#
# to_train = lgb.Dataset(X_train_d, y_train)
# to_val = lgb.Dataset(X_test_d, y_test)
# model_lgb = lgb.train(params,
#                       to_train,
#                       valid_sets=[to_train, to_val],
#                       num_boost_round=10000,
#                       verbose_eval=100,
#                       early_stopping_rounds=100)
# p_train = model_lgb.predict(X_train_d)
# p_test = model_lgb.predict(X_test_d)
#
# accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
# roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]
#
# print(accuracy)
# print(roc_auc)



# pca = PCA(n_components=1000,
#           random_state=228)
# pca.fit(X_train_d.toarray())
# X_train_d = pca.transform(X_train_d.toarray())
# X_test_d = pca.transform(X_test_d.toarray())

model_perceptron = Perceptron(max_iter=200,
                              penalty=None,
                              early_stopping=True,
                              validation_fraction=0.05,
                              random_state=228)
model_perceptron.fit(X_train_f, y_train)
p_train = model_perceptron.predict(X_train_f)
p_test = model_perceptron.predict(X_test_f)

accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]

print(accuracy)
print(roc_auc)


model_logit = LogisticRegression(penalty='l1',
                                 solver='liblinear',
                                 C=1,
                                 tol=1e-3,
                                 max_iter=100,
                                 random_state=228)
model_logit.fit(X_train_f, y_train)
p_train = model_logit.predict(X_train_f)
p_test = model_logit.predict(X_test_f)

accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]

print(accuracy)
print(roc_auc)


model_forest = RandomForestClassifier(n_estimators=100,
                                      oob_score=True,
                                      max_depth=100,
                                      min_samples_leaf=1,
                                      verbose=1,
                                      random_state=228)
model_forest.fit(X_train_f, y_train)
p_train = model_forest.predict(X_train_f)
p_test = model_forest.predict(X_test_f)

accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]

print(accuracy)
print(roc_auc)

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

to_train = lgb.Dataset(X_train_f, y_train)
to_val = lgb.Dataset(X_test_f, y_test)
model_lgb = lgb.train(params,
                      to_train,
                      valid_sets=[to_train, to_val],
                      num_boost_round=10000,
                      verbose_eval=100,
                      early_stopping_rounds=100)
p_train = model_lgb.predict(X_train_f)
p_test = model_lgb.predict(X_test_f)

accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]

print(accuracy)
print(roc_auc)
