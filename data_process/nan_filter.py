import pandas as pd
from data_process.parameters import cat, factor_list
import numpy as np


def data_filter(data):
    ave = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    data = (data - ave) / std
    return data


data1 = pd.read_csv(open('../' + cat + '/data.csv'), index_col=0, header=0)
data1.dropna(axis=0, how='all', inplace=True)
data1.dropna(axis=1, how='all', inplace=True)
data1.dropna(axis=0, thresh=0.5 * data1.shape[1], inplace=True)
data1.dropna(axis=1, thresh=0.5 * data1.shape[0], inplace=True)
data1 = data1.values.astype(float)
data1 = data_filter(data1)

data1 = pd.DataFrame(data1)
data1.fillna(0, inplace=True)
data1.to_csv('../' + cat + '/data_filtered.csv', encoding='utf-8')
