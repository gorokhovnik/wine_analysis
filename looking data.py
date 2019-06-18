import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from FE import FE
from RS import RS

wine = pd.read_csv('wine.csv')

wine = RS(wine)
wine = FE(wine, 500)

cols = list(wine.columns)

print(cols)
print()

#
# '''страны'''
# print('- country - страна производства\n\n![](report_img/countries_pie.png)\n')
# countries = wine[['country', 'title']].groupby('country').count().sort_values('title', ascending=False)
# countries_count = countries['title'].tolist()
# countries = countries.index.tolist()
# print('|Country|Count|\n|---|---|')
# for idx, country in enumerate(countries):
#     if idx == 10:
#         break
#     if country != 'Other':
#         print('|' + country + '|' + str(countries_count[idx]) + '|')
# print()
#
# plt.pie(countries_count, labels=countries)
# plt.savefig('report_img/countries_pie.png')
# plt.show()
#
# print('Страны, в которых производится менее 500 видов вина объединены в группу Other\n\n- continent - континент производства')
#
# '''континенты'''
# print('\n![](report_img/continents_pie.png)\n')
# continents = wine[['continent', 'title']].groupby('continent').count().sort_values('title', ascending=False)
# continents_count = continents['title'].tolist()
# continents = continents.index.tolist()
# print('|Continent|Count|\n|---|---|')
# for idx, continent in enumerate(continents):
#     print('|' + continent + '|' + str(continents_count[idx]) + '|')
# print('\n')
#
# plt.pie(continents_count, labels=continents)
# plt.savefig('report_img/continents_pie.png')
# plt.show()

'''цвет'''
print('- color - цвет вина\n\n![](report_img/color_bar.png)\n')
colors = wine[['color', 'title']].groupby('color').count()
colors_count = colors['title'].tolist()
colors = colors.index.tolist()
print('|Color|Count|\n|---|---|')
for idx, color in enumerate(colors):
    print('|' + color + '|' + str(colors_count[idx]) + '|')
print('\nПри примерно одинаковой средней оценке белые вина заметно дешевле красных, поэтому будет целесообразно рассматривать вина по-отдельности\n')

ax = wine[['color', 'title']].groupby('color').count().plot(kind='bar', color='black')
wine[['color', 'points', 'price']].groupby('color').mean().plot(secondary_y=True, xlim=ax.get_xlim(), ax=ax, color=['red', 'blue'], linewidth=2, markersize=10)
ax.legend().remove()
plt.legend(loc='center')
plt.savefig('report_img/color_bar.png')
plt.show()
print(colors)

