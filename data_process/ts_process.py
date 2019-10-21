import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tsfresh import extract_relevant_features

cat = '情绪类'

df = pd.read_csv(open('./' + cat + '/data.csv'), header=0, index_col=0)
data = df.values
data = data.reshape([data.shape[0], -1, 300, 15])
data = data.transpose([1, 2, 3, 0])
data = data.reshape([data.shape[0], -1, data.shape[3]])
data = np.delete(data, [10, 11, 19, 28, 47], axis=2)

ids = []
times = []
for i in range(300):
    for j in range(15):
        ids.append(i)
        times.append(j)
ids = np.array(ids).T
times = np.array(times).T
for i in range(data.shape[0]):
    dat = pd.DataFrame(data[i])
    dat['id'] = ids
    dat['time'] = times
    dat.to_csv('./' + cat + '/' + str(i) + '.csv')

month = 0
dat = pd.read_csv(open('./' + cat + '/' + str(month) + '.csv'))
dat = dat.dropna()
dat[dat['id'] == 5].plot(subplots=True, sharex=True, figsize=(20, 100))
X = extract_relevant_features(dat, column_id="id", column_sort="time")
