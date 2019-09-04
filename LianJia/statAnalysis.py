import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt

df = pd.read_csv('D:/Codes/PythonSkills/LianJia/contents.csv', sep = '\t')
# 0 1 2 3 4 5 6     7     8    9      10  11  12         13      14   15      16      17
# E S W N y x space floor time subway new key decoration heating rent twobath deposit price
plt.scatter(df['E'], df['price'], s = 1)
plt.show()