import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from FE import FE
from RS import RS

warnings.filterwarnings('ignore')

wine = pd.read_csv('wine.csv')

RS(wine)
FE(wine)

cols = list(wine.columns)
print(cols)

'''цена/качество'''
wine['points_to_price'] = wine['points'] / wine['price']
print('|||||\n|---|---|---|---|')
for color in ['red', 'white']:
    for category in range(1, 5):
        idx = wine[(wine['category'] == category) & (wine['color'] == color)]['points_to_price'].argmax()
        print('|' + wine.loc[idx]['title'] + ' ' + str(wine.loc[idx]['price']) + '$ ' + str(wine.loc[idx]['points']) + 'p', end='')
    print('|')

# print(wine.sort_values('points_to_price', ascending=False)['title'])