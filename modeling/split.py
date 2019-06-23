import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

def split(wine, description='description', features=[], y='', test_size=0.2, tfidf=None):
    X = wine[[description] + features]
    y = wine[y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=228)

    X_d = X[description].str.lower().replace('[^a-zA-Z0-9%]', ' ', regex=True).replace('  ', ' ', regex=True)
    X_f = pd.get_dummies(X[features])

    X_train_d = X_train[description].str.lower().replace('[^a-zA-Z0-9%]', ' ', regex=True).replace('  ', ' ', regex=True)
    X_train_f = pd.get_dummies(X_train[features])

    X_test_d = X_test[description].str.lower().replace('[^a-zA-Z0-9%]', ' ', regex=True).replace('  ', ' ', regex=True)
    X_test_f = pd.get_dummies(X_test[features])

    if tfidf is None:
        tfidf = TfidfVectorizer()

    tfidf.fit(X_train_d)
    X_train_d = tfidf.fit_transform(X_train_d)
    X_test_d = tfidf.transform(X_test_d)
    X_d = tfidf.transform(X_d)

    X_train_f = pd.get_dummies(X_train_f)
    X_test_f = pd.get_dummies(X_test_f)
    X_f = pd.get_dummies(X_f)

    X = hstack([X_d, X_f])
    X_train = hstack([X_train_d, X_train_f])
    X_test = hstack([X_test_d, X_test_f])

    return X, X_d, X_f, y, X_train, X_train_d, X_train_f, y_train, X_test, X_test_d, X_test_f, y_test
