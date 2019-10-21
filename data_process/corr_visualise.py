import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_process.parameters import cat

data = pd.read_csv(open('../' + cat + '/data_filtered.csv'), index_col=0, header=0).values
delete_list = np.array([True] * data.shape[0])

corr = np.corrcoef(data)
plt.figure(figsize=[19.2, 10.8])
sns.heatmap(corr, annot=True)
plt.savefig('../' + cat + '/corr.png')
