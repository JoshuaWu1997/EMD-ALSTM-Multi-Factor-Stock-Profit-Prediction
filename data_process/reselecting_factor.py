import pandas as pd
import numpy as np
from skfeature.function.information_theoretical_based import MIFS
from parameters import cat
from factor_list import get_infomation

factor_list, select_list, back = get_infomation(cat)
factor_list = np.array(factor_list)

f_data = 'G:/factor_db/' + cat + '/data.csv'
k_data = './kdata.csv'

fd = pd.read_csv(open(f_data), index_col=0, header=0)
fd.index = factor_list
fd = fd.T[select_list].T

kd = pd.read_csv(k_data, index_col=0, header=0)
fd = fd.values.reshape(fd.shape[0], -1, 300, 15).transpose((1, 2, 0, 3))  # month,stock,factor,time#

kd = kd.values.transpose((1, 0))
print(fd.shape, kd.shape)

select_factor = dict()
for month in range(fd.shape[0]):
    filter0 = np.logical_not(
        np.logical_or(
            np.isinf(np.ravel(kd[month])), np.isnan(np.ravel(kd[month]))
        )
    )
    kdata = np.ravel(kd[month])[filter0]
    fdata = fd[month][filter0]
    print(fdata.shape, kdata.shape)

    fdata = np.nanmean(fdata, axis=2)
    nan_num = np.sum(np.isnan(fdata), axis=0)
    filter1 = nan_num < 0.3 * fdata.shape[0]
    factor_name = np.array(select_list)
    fdata = fdata[:, filter1]
    factor_name = factor_name[filter1]
    print(fdata.shape, kdata.shape)

    filter0 = np.sum(np.isnan(fdata), axis=1) == 0
    fdata = fdata[filter0]
    kdata = kdata[filter0]
    print(fdata.shape, kdata.shape)

    F, J_CMI, MIfy = MIFS.mifs(fdata, kdata)
    select = factor_name[F[:]]
    print(select)
    for j in range(len(select)):
        select_factor[select[j]] = select_factor.get(select[j], 0) + 1

select_factor = pd.DataFrame([list(select_factor.keys()), list(select_factor.items())], index=['name', 'freq'])
select_factor.sort_values(by='freq', axis=1, ascending=False)
print('\n', select_factor.values)
