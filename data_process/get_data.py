# -*- coding: utf-8 -*-

import os
from atrader import *
import numpy as np
import pandas as pd
from data_process.parameters import cat, factor_list, back


def get_fd(context):
    data = get_reg_factor(reg_idx=context.reg_factor[0], target_indices=(), length=back, df=True)
    data = data['value'].values.reshape([-1, len(factor_list), back]).astype(float)
    # (factor, stocks, times)
    data = np.swapaxes(data, 0, 1)
    data = data.reshape([len(factor_list), -1])
    data = pd.DataFrame(data)
    return data


def init(context):
    set_backtest(initial_cash=1e6, stock_cost_fee=30)
    reg_factor(factor_list)
    context.month = 0
    if not os.path.exists('../' + cat):
        os.makedirs('../' + cat)


def on_data(context):
    data = get_fd(context)
    if context.month > 0:
        pre_data = pd.read_csv(open('../' + cat + '/data.csv'), index_col=0, header=0)
        data = pd.concat([pre_data, data], axis=1)
    data.to_csv('../' + cat + '/data.csv')
    context.month += 1


if __name__ == '__main__':
    # TIME MUST BE DURING: 2016-01-01 to 2018-09-30
    run_backtest(strategy_name='fac_demo', file_path='get_data.py', target_list=get_code_list('hs300')['code'],
                 frequency='month', fre_num=1, begin_date='2016-01-01', end_date='2018-09-30', fq=1)
