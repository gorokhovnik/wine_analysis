import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score

from FE import FE
from RS import RS

wine = pd.read_csv('../../wine.csv')
wine = FE(wine, 500)
wine = RS(wine)
wine_summary = pd.read_csv('../../wine_summary.csv')
wine_summary.set_index('id', inplace=True)

X = wine[['description', 'country', 'continent', 'price', 'category', 'year']]
y = wine['color'].map({'red': 0, 'white': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=228)

X_train_d = X_train['description'].str.lower().replace('[^a-zA-Z0-9%]', ' ', regex=True).replace('  ', ' ', regex=True)
X_train_f = X_train[['country', 'continent', 'price', 'category', 'year']]

X_test_d = X_test['description'].str.lower().replace('[^a-zA-Z0-9%]', ' ', regex=True).replace('  ', ' ', regex=True)
X_test_f = X_test[['country', 'continent', 'price', 'category', 'year']]

accuracy = []
roc_auc = []

tfidf = TfidfVectorizer(min_df=20)
X_train_d = tfidf.fit_transform(X_train_d)
X_test_d = tfidf.transform(X_test_d)

model_perceptron = Perceptron()
model_perceptron.fit(X_train_d, y_train)
p_train = model_perceptron.predict(X_train_d)
p_test = model_perceptron.predict(X_test_d)

accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]

model_forest = RandomForestClassifier(100)
model_forest.fit(X_train_d, y_train)
p_train = model_forest.predict(X_train_d)
p_test = model_forest.predict(X_test_d)

accuracy += [[accuracy_score(y_train, p_train), accuracy_score(y_test, p_test)]]
roc_auc += [[roc_auc_score(y_train, p_train), roc_auc_score(y_test, p_test)]]

print(accuracy)
print(roc_auc)