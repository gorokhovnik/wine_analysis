import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
wine_summary = pd.read_csv('../../wine_summary.csv')
wine_summary.set_index('id', inplace=True)

X_d, X_f, y,\
X_train_d, X_train_f, y_train,\
X_test_d, X_test_f, y_test = split(wine, 'description', ['country', 'continent', 'price', 'category', 'year'], 'color')

accuracy = []
roc_auc = []

tfidf = TfidfVectorizer(min_df=20,
                        ngram_range=(1, 1),
                        norm='l2',
                        max_features=5000)
X_train_d = tfidf.fit_transform(X_train_d)
X_test_d = tfidf.transform(X_test_d)

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
model_perceptron.fit(X_train_d, y_train)
p_train = model_perceptron.predict(X_train_d)
p_test = model_perceptron.predict(X_test_d)

accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]

print(accuracy)
print(roc_auc)

model_logit = LogisticRegression(random_state=228)
model_logit.fit(X_train_d, y_train)
p_train = model_logit.predict(X_train_d)
p_test = model_logit.predict(X_test_d)

accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]

print(accuracy)
print(roc_auc)

model_forest = RandomForestClassifier(100, random_state=228)
model_forest.fit(X_train_d, y_train)
p_train = model_forest.predict(X_train_d)
p_test = model_forest.predict(X_test_d)

accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]

print(accuracy)
print(roc_auc)

params = {
    'max_bin': 7,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 0.04,
    'learning_rate': 0.0085,
    'max_depth': 50,
    'metric': 'accuracy',
    'min_data_in_leaf': 3,
    'min_sum_hessian_in_leaf': 10,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary'
}

# to_train = lgb.Dataset(X_train_d.values, y_train.values)
# to_val = lgb.Dataset(X_test_d.values, y_test.values)
# model_lgb = lgb.train(params, to_train, valid_sets=[to_train, to_val], num_boost_round=1000, verbose_eval=100)
# p_train = model_lgb.predict(X_train_d)
# p_test = model_lgb.predict(X_test_d)

model_boost = XGBClassifier(5, 0.05, 10, 0, random_state=228)
model_boost.fit(X_train_d, y_train)
p_train = model_boost.predict(X_train_d)
p_test = model_boost.predict(X_test_d)

accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]

print(accuracy)
print(roc_auc)
