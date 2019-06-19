import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from FE import FE
from RS import RS

warnings.filterwarnings('ignore')

wine = pd.read_csv('wine.csv')

wine = FE(wine, 500)
wine = RS(wine)

cols = list(wine.columns)
print(cols)
print('\n')

'''цена/качество'''
print('Лучшие вина:\n')
print('|||||\n|---|---|---|---|')
for color in ['red', 'white']:
    for category in range(1, 5):
        idx = wine[(wine['category'] == category) & (wine['color'] == color)]['points'].argmax()
        print('|' + wine.loc[idx]['title'] + ' ' + str(wine.loc[idx]['price']) + '$ ' + str(
            wine.loc[idx]['points']) + 'p', end='')
    print('|')
print()
wine_summary = wine[['title', 'price', 'points']]

print('Лучшие вина по отношению оценка / цена:\n')
wine['points_to_price'] = wine['points'] / wine['price']
print('|||||\n|---|---|---|---|')
for color in ['red', 'white']:
    for category in range(1, 5):
        idx = wine[(wine['category'] == category) & (wine['color'] == color)]['points_to_price'].argmax()
        print('|' + wine.loc[idx]['title'] + ' ' + str(wine.loc[idx]['price']) + '$ ' + str(
            wine.loc[idx]['points']) + 'p', end='')
    print('|')
print()
wine_summary['points_to_price'] = wine['points_to_price']

print('Лучшие цены при изменении оценки по формуле new_points = (points – 80)**2:\n')
wine['points_to_price'] = (wine['points'] - 80) ** 2 / wine['price']
print('|||||\n|---|---|---|---|')
for color in ['red', 'white']:
    for category in range(1, 5):
        idx = wine[(wine['category'] == category) & (wine['color'] == color)]['points_to_price'].argmax()
        print('|' + wine.loc[idx]['title'] + ' ' + str(wine.loc[idx]['price']) + '$ ' + str(
            wine.loc[idx]['points']) + 'p', end='')
    print('|')
wine_summary['norm_points_to_price'] = wine['points_to_price']
print('\n')

'''лучшие страны'''
countries = wine[['country', 'price', 'points', 'points_to_price']].groupby('country').mean()
print(countries)

wine_summary.to_csv('wine_summary.csv')
