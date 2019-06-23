import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import vstack
import warnings
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

from FE import FE
from RS import RS
from modeling.split import split

warnings.filterwarnings('ignore')

wine = pd.read_csv('wine.csv')

wine = FE(wine, 500)
wine = RS(wine)

wine_summary = pd.read_csv('wine_summary.csv')
wine_summary.set_index('id', inplace=True)

tfidf = TfidfVectorizer(min_df=20,
                        ngram_range=(1, 1),
                        norm='l2',
                        max_features=5000)

X, X_d, X_f, y, \
X1, X1_d, X1_f, y1, \
X2, X2_d, X2_f, y2 = split(wine=wine,
                           description='description',
                           features=['country', 'continent', 'price', 'category', 'color', 'year'],
                           y='points',
                           test_size=0.5,
                           tfidf=tfidf)

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

print('lgb on 1st')
to_train = lgb.Dataset(X1, y1)
to_val = lgb.Dataset(X2, y2)
model_lgb = lgb.train(params,
                      to_train,
                      valid_sets=[to_train, to_val],
                      num_boost_round=2000,
                      verbose_eval=100,
                      early_stopping_rounds=100)
p2 = model_lgb.predict(X2)

print('lgb on 2st')
to_train = lgb.Dataset(X2, y2)
to_val = lgb.Dataset(X1, y1)
model_lgb = lgb.train(params,
                      to_train,
                      valid_sets=[to_train, to_val],
                      num_boost_round=2000,
                      verbose_eval=100,
                      early_stopping_rounds=100)
p1 = model_lgb.predict(X1)
points1 = pd.DataFrame({'y': y1, 'p': np.floor(p1)})
points2 = pd.DataFrame({'y': y2, 'p': np.floor(p2)})

points = pd.concat([points1, points2])
points['p'] = points['p'].astype('int32')

wine_summary = wine_summary.join(points['p'])
wine_summary['p_diff'] = wine_summary['p'] - wine_summary['points']

wine_summary = wine_summary.join(wine['category'])
wine_summary = wine_summary.join(wine['color'])

wine_summary.to_csv('wine_summary_ML.csv')

print('Лучшие вина по разнице ожидаемых и реальных очков:\n')
print('|||||\n|---|---|---|---|')
for color in range(2):
    for category in range(4):
        idx = wine_summary[(wine_summary['category'] == category) & (wine_summary['color'] == color)]['p_diff'].argmax()
        print('|' + wine.loc[idx]['title'] + ' ' + str(wine.loc[idx]['price']) + '$ ' + str(
            wine.loc[idx]['points']) + 'p ' + str(wine_summary.loc[idx]['p']) + 'p pred', end='')
    print('|')
print('\n')

print('lgb on full')
to_train = lgb.Dataset(X, y)
model_lgb = lgb.train(params,
                      to_train,
                      valid_sets=[to_train],
                      num_boost_round=4000,
                      verbose_eval=1000)
model_lgb.save_model('points_model')
