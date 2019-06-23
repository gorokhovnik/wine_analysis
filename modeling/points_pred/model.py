import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from FE import FE
from RS import RS
from modeling.split import split

wine = pd.read_csv('../../wine.csv')

wine = FE(wine, 500)
wine = RS(wine)

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
                                           features=['country', 'continent', 'price', 'category', 'color', 'year'],
                                           y='points',
                                           test_size=0.5,
                                           tfidf=tfidf)

MAEs = []

print('lgb on full')
params = {
    'max_bin': 127,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 0.85,
    'bagging_fraction': 0.8,
    'learning_rate': 0.05,
    'max_depth': 25,
    'metric': 'mae',
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

MAEs += [[mean_absolute_error(y_train, p_train), mean_absolute_error(y_test, p_test)]]

print(MAEs)
model_lgb.save_model('full')