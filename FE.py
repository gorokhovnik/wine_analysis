import pandas as pd
import numpy as np

def FE(wine):
    wine.set_index('id', inplace=True)
    wine['year'] = wine['title'].str.extract(r'[0-9]')