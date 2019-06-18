import pandas as pd
import numpy as np

def RS(wine):
    wine.dropna(inplace=True)
    wine = wine[wine['price'] <= 100]
