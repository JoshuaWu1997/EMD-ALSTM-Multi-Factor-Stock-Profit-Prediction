# -*- coding: utf-8 -*-
from atrader import *
import numpy as np
from EMD_LSTM import EMD_LSTMA, strategy_name
import warnings
from func import get_filter, prediction_cross_validate
from func import factor_list
import sys

warnings.filterwarnings('ignore')

sys.stdout = open(strategy_name + '.txt', 'a')
result_fname = strategy_name + '.csv'
order_fname = strategy_name + '_order.csv'


def get_order(predict):
    pred = predict[predict != -1]
    ave = np.mean(pred)
    std = np.std(pred)
    long = len(pred[pred > 0])
    print('predict mean and std:\t', [ave, std], '\nratio:\t', np.sqrt(long / len(pred)))
    zeros = np.logical_and(predict > np.max(ave, 0), predict < ave + 2 * std)
    minus1 = predict < 0
    predict[zeros] = 0
    predict[minus1] = -1
    return predict, np.sqrt(long / len(pred))


def get_fd(context):
    data = get_reg_factor(reg_idx=context.reg_factor[0], target_indices=(), length=context.back_date, df=True)
    data = data['value'].values.reshape([-1, len(factor_list), context.back_date]).astype(float)
    data = data[:, :, -15:]
    return data


def get_pr(context):
    data = get_reg_kdata(reg_idx=context.reg_kdata[0], target_indices=(), length=1, df=True)
    open_p = data.open.values.reshape(-1, 1).astype(float)
    close_p = data.close.values.reshape(-1, 1).astype(float)
    label = (close_p - open_p) / open_p
    return label


def init(context):
    set_backtest(initial_cash=1e7, stock_cost_fee=30)
    reg_factor(factor_list)
    reg_kdata('month', 1)

    context.Tlen = len(context.target_list)
    context.month = 0
    context.pre_predict = np.array([0])
    context.f_data = np.array([0])
    context.k_label = np.array([0])
    context.back_date = 30


def on_data(context):
    print('\n############## MONTH', context.month, '#################')
    if context.month == 0:
        context.f_data = get_fd(context)
        # pd.DataFrame(context.target_list).to_csv(result_fname)
        # pd.DataFrame(context.target_list).to_csv(order_fname)
    else:
        # ------------------------------Initiate--------------------------------#
        positions = context.account().positions['volume_long'].values
        valid_cash = context.account().cash['valid_cash'].values
        total_asset = context.account().cash['total_asset'].values
        total_value = context.account().cash['total_value'].values
        print('total_asset:\t', total_asset, 'total_value:\t', total_value)
        target = np.array(range(context.Tlen))
        # ------------------------------GET DATA--------------------------------#
        f_data = get_fd(context)
        k_label = get_pr(context)
        model_acc = prediction_cross_validate(k_label, context.pre_predict)
        print('MODEL ACCURACY:\t', model_acc)
        # ------------------------------Data Filter-----------------------------#
        target_filter = get_filter(f_data)
        train_filter = get_filter(context.f_data, k_label)
        # ------------------------------MODEL FITTING(FUNCTION)-----------------#
        model = EMD_LSTMA(train_data=context.f_data, train_label=k_label, fit_data=f_data,
                          target_filter=target_filter, train_filter=train_filter, split=10,
                          month=context.month, method='regression')
        model.fit_predict()
        predict = model.predict  # one-dimension array as weights of stocks
        predict = np.ravel(predict)
        # ------------------------------SETTING ORDER (parameters: predict)-----#
        # result = pd.read_csv(result_fname, header=0, index_col=0)
        # result[str(context.month)] = predict
        # result.to_csv(result_fname)
        context.pre_predict = predict
        context.f_data = f_data
        # -------------------------------ORDER----------------------------------#
        predict, ratio = get_order(predict)
        # result = pd.read_csv(order_fname, header=0, index_col=0)
        # result[str(context.month)] = predict
        # result.to_csv(order_fname)
        long = np.logical_and(positions == 0, predict > 0)
        short = np.logical_and(positions > 0, predict < 0)
        target_long, target_short = target[long].tolist(), target[short].tolist()
        value = np.sum(predict[long])
        print('valid cash\t', valid_cash[0])
        print('short_list\t', len(target_short))
        for targets in target_short:
            order_target_volume(account_idx=0, target_idx=targets, target_volume=0, side=1,
                                order_type=2, price=0)
        if value != 0:
            unit = 1e7 * ratio / value
            print('unit:\t', unit)
            order = unit * np.ravel(predict[long])
            print('long_list\t', len(target_long))
            for i in range(len(target_long)):
                order_target_value(account_idx=0, target_idx=target_long[i], target_value=order[i], side=1,
                                   order_type=2, price=0)
        # -------------------------------------------------------------------------
    context.month += 1


if __name__ == '__main__':
    run_backtest(strategy_name=strategy_name, file_path='factor.py', target_list=get_code_list('hs300')['code'],
                 frequency='month', fre_num=1, begin_date='2016-01-01', end_date='2018-09-30', fq=1)
