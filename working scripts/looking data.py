import pandas as pd
import numpy as np

wine = pd.read_csv('../wine.csv')
print(wine.dropna().info())
print(wine.info())