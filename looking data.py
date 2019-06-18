import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from FE import FE
from RS import RS

wine = pd.read_csv('wine.csv')

RS(wine)
FE(wine)

cols = list(wine.columns)

print(cols)

sns.jointplot(x='price', y='category', data=wine, kind="kde")
plt.show()