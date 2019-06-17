import re
import pandas as pd
import numpy as np

red = []
white = []
rose = []

raw_red = pd.read_csv('red.csv')
raw_white = pd.read_csv('white.csv')
raw_rose = pd.read_csv('rose.csv')


for idx, row in raw_red.iterrows():
    red += row['Common Name(s)'].split('/')
    if not pd.isna(row['All Synonyms']):
        red += row['All Synonyms'].split(',')
for idx, word in enumerate(red):
    red[idx] = re.sub('[\(\[].*?[\)\]]', '', red[idx])
    if red[idx][0] == ' ':
        red[idx] = red[idx][1:]
    if red[idx][0] == ' ':
        red[idx] = red[idx][1:]
    if red[idx][-1] == ' ' or red[idx][-1] == '.':
        red[idx] = red[idx][:-1]
red.sort()

for idx, row in raw_white.iterrows():
    white += row['Common Name(s)'].split('/')
    if not pd.isna(row['All Synonyms']):
        white += row['All Synonyms'].split(',')
for idx, word in enumerate(white):
    white[idx] = re.sub('[\(\[].*?[\)\]]', '', white[idx])
    if white[idx] != '':
        if white[idx][0] == ' ':
            white[idx] = white[idx][1:]
        if white[idx][0] == ' ':
            white[idx] = white[idx][1:]
        if white[idx][-1] == ' ' or white[idx][-1] == '.':
            white[idx] = white[idx][:-1]
white.sort()
white = white[1:][:-2]

for idx, row in raw_rose.iterrows():
    rose += row['Common Name(s)'].split('/')
    if not pd.isna(row['All Synonyms']):
        rose += row['All Synonyms'].split(',')
for idx, word in enumerate(rose):
    rose[idx] = re.sub('[\(\[].*?[\)\]]', '', rose[idx])
    if rose[idx][0] == ' ':
        rose[idx] = rose[idx][1:]
    if rose[idx][0] == ' ':
        rose[idx] = rose[idx][1:]
    if rose[idx][-1] == ' ' or rose[idx][-1] == '.':
        rose[idx] = rose[idx][:-1]

rose.sort()


print('red =', red)
print('white =', white)
print('rose =', rose)