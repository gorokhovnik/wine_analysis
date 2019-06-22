import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split(wine, description='description', features=[], y='', test_size=0.2):
    X = wine[[description] + features]
    y = wine[y].map({'red': 0, 'white': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=228)

    X_d = X[description].str.lower().replace('[^a-zA-Z0-9%]', ' ', regex=True).replace('  ', ' ', regex=True)
    X_f = X[features]

    X_train_d = X_train[description].str.lower().replace('[^a-zA-Z0-9%]', ' ', regex=True).replace('  ', ' ', regex=True)
    X_train_f = X_train[features]

    X_test_d = X_test[description].str.lower().replace('[^a-zA-Z0-9%]', ' ', regex=True).replace('  ', ' ', regex=True)
    X_test_f = X_test[features]

    del X
    del X_train
    del X_test

    return X_d, X_f, y, X_train_d, X_train_f, y_train, X_test_d, X_test_f, y_test
