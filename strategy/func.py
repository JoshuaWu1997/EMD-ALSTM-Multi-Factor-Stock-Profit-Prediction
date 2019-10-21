import numpy as np
import pandas as pd
from PyEMD import EMD
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction.settings import from_columns
from tsfresh import extract_relevant_features, extract_features
import warnings

warnings.filterwarnings("ignore")
factor_list = ['Skewness20', 'MA10', 'RSTR42', 'VROC6', 'Kurtosis20', 'CR20', 'CCI5', 'ILLIQUIDITY', 'VOL10', 'MFI',
               'RSTR21',

               'IntCL', 'NonCurrentAssetsRatio', 'PS', 'DebtEquityRatio', 'ValueChgProfit', 'BondsPayableToAsset', 'PE',
               'RetainedEarnings', 'NegMktValue', 'BasicEPS']

'''
factor_list = ['Variance20', 'Kurtosis20', 'LossVariance20', 'GainVariance20', 'Treynorratio20', 'VSTD10', 'VOL10',
               'VSTD20', 'VROC6', 'MA10', 'DHILO', 'MFI', 'CR20', 'ILLIQUIDITY', 'CCI5', 'BULLPOWER', 'RSTR21', ]

factor_list = ['NetWorkingCapital', 'NetDebt', 'RetainedEarnings', 'IntCL', 'ValueChgProfit', 'NegMktValue', 'PE',
               'MktValue', 'StaticPE', 'FinancingCashGrowRate', 'NPParentCompanyGrowRate', 'OperCashGrowRate',
               'NetProfitGrowRate']

factor_list = [sys.argv[1]]
print(factor_list)
'''


def emd_data_transform(data):
    samples, time_steps, data_dims = data.shape
    data_mf1 = []
    data_mf2 = []
    for i in range(samples):
        sample_1 = []
        sample_2 = []
        for j in range(data_dims):
            S = np.ravel(data[i, :, j])
            emd = EMD()
            IMFs = emd(S)
            sample_1.append(IMFs[0].tolist())
            sample_2.append((S - IMFs[0]).tolist())
        data_mf1.append(sample_1)
        data_mf2.append(sample_2)
    data_mf1 = np.array(data_mf1).transpose([0, 2, 1])
    data_mf2 = np.array(data_mf2).transpose([0, 2, 1])

    return [data_mf1, data_mf2]


def get_filter(data, label=None):
    data = data.reshape([data.shape[0], -1]).T
    data_indicator = np.isnan(data)
    data_missing = np.sum(data_indicator, axis=0)

    data_indicator = data_missing > 0
    if label is not None:
        data_indicator = np.logical_or(data_indicator,
                                       np.logical_or(np.ravel(np.isnan(label)), np.ravel(np.isinf(label))))

    data_indicator = np.logical_not(data_indicator)
    return data_indicator.tolist()


def prediction_cross_validate(y_true, y_pred):
    if len(y_pred) > 1:
        flt = np.ravel(np.logical_not(
            np.logical_or(np.isnan(y_true), np.isinf(y_true))
        ))
        y_true, y_pred = to_category(y_true[flt]), to_category(y_pred[flt])
        print('prediction report:\n', classification_report(y_true, y_pred))
        print('confusion matrix:\n', confusion_matrix(y_true, y_pred))
        return accuracy_score(y_true, y_pred)


def to_category(label):
    label = np.where(label < 0, -1, label)
    label = np.where(label > 0, 1, label)
    label = np.where(abs(label) != 1, 0, label)
    return np.ravel(label)


def get_features(X, y=None, kind_to_fc_parameters=None):
    samples, time_steps, data_dim = X.shape
    X = X.reshape([-1, data_dim])
    time = list(range(time_steps)) * samples
    ids = []
    for i in range(samples):
        ids.extend([i] * time_steps)
    X = pd.DataFrame(X)
    X['id'] = ids
    X['time'] = time

    if y is not None:
        features = extract_relevant_features(X, y, column_id='id', column_sort='time', n_jobs=0)
        kind_to_fc_parameters = from_columns(features)
        return features.values, kind_to_fc_parameters
    elif kind_to_fc_parameters is not None:
        features = extract_features(X, column_id='id', column_sort='time', n_jobs=0,
                                    default_fc_parameters=kind_to_fc_parameters)
    else:
        features = extract_features(X, column_id='id', column_sort='time', n_jobs=0)
    return features.values


def data_difference(data):
    data = data[:, :, 1:] - data[:, :, :-1]
    return data
