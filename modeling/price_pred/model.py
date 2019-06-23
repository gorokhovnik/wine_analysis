import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

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


X, X_d, X_f, y, \
X_train, X_train_d, X_train_f, y_train, \
X_test, X_test_d, X_test_f, y_test = split(wine=wine,
                                           description='description',
                                           features=['country', 'continent', 'color', 'year'],
                                           y='price',
                                           tfidf=tfidf)

MSEs = []


print('lgb on description')
params = {
    'max_bin': 7,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 0.85,
    'bagging_fraction': 0.8,
    'learning_rate': 0.05,
    'max_depth': 25,
    'metric': 'mse',
    'min_data_in_leaf': 1,
    'min_sum_hessian_in_leaf': 0,
    'num_leaves': 101,
    'num_threads': 8,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'tree_learner': 'serial',
    'objective': 'regression'
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

MSEs += [[mean_squared_error(y_train, p_train_d), mean_squared_error(y_test, p_test_d)]]

print(MSEs)
model_lgb.save_model('description')


print('lgb on feat')
params = {
    'max_bin': 255,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 1,
    'bagging_fraction': 0.8,
    'learning_rate': 0.05,
    'max_depth': 25,
    'metric': 'mse',
    'min_data_in_leaf': 1,
    'min_sum_hessian_in_leaf': 0,
    'num_leaves': 101,
    'num_threads': 8,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'tree_learner': 'serial',
    'objective': 'regression'
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

MSEs += [[mean_squared_error(y_train, p_train_f), mean_squared_error(y_test, p_test_f)]]

print(MSEs)
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
    'metric': 'mse',
    'min_data_in_leaf': 1,
    'min_sum_hessian_in_leaf': 0,
    'num_leaves': 101,
    'num_threads': 8,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'tree_learner': 'serial',
    'objective': 'regression'
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

MSEs += [[mean_squared_error(y_train, p_train), mean_squared_error(y_test, p_test)]]

print(MSEs)
model_lgb.save_model('full')