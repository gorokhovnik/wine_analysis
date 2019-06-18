import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

wine = pd.read_csv('../wine.csv')

sns.heatmap(wine.isnull().T, xticklabels=False, cbar=False).set_title('missing')
plt.savefig('../report_img/missing.png')
plt.show()