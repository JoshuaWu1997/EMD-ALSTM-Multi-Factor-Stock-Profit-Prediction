from strategy import StrategyBase
from keras.layers import *
from keras.models import *
from keras.utils import to_categorical
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from layers import stacked_dense
from keras_func import reduce_lr, early_stop, csv_log
import gc

strategy_name = 'MLP'
factor = [['Variance20', 'Kurtosis20', 'Kurtosis60', 'Skewness20'],
          ['VSTD10', 'VOL10', 'VSTD20', 'VROC6'],
          ['MA10', 'DHILO', 'MFI', 'CR20', 'ILLIQUIDITY'],
          ['CCI5', 'BULLPOWER', 'RSTR42', 'RSTR21'],
          ['NetWorkingCapital', 'NetDebt', 'RetainedEarnings', 'IntCL', 'ValueChgProfit'],
          ['DebtEquityRatio', 'SuperQuickRatio', 'NonCurrentAssetsRatio', 'BondsPayableToAsset'],
          ['FinancingCashGrowRate', 'NPParentCompanyGrowRate', 'OperCashGrowRate', 'NetProfitGrowRate'],
          ['NegMktValue', 'StaticPE', 'MktValue', 'PS'],
          ['NetAssetPS', 'TORPS', 'BasicEPS', 'DividendPS']]


def PowerSetsRecursive(items):
    # the power set of the empty set has one element, the empty set
    result = [[]]
    for x in items:
        result.extend([subset + [x] for subset in result])
    return result[:-1]


def model_mlp_dynamic(time_steps, data_dim, num_classes, method='category'):
    inputs = []
    denses = []
    for i in range(data_dim):
        input = Input(shape=(time_steps,))
        dense = Dense(time_steps, activation='relu')(input)
        dense_norm = BatchNormalization()(dense)
        inputs.append(input)
        denses.append(dense_norm)
    if data_dim > 1:
        mux = Concatenate()(denses)
    else:
        mux = denses[0]
    dense = Dense(data_dim * 8, activation='tanh')(mux)
    dense = Dense(data_dim * 8, activation='tanh')(dense)

    if method == 'category':
        output = Dense(num_classes, activation='softmax')(dense)
    else:
        x = Dense(num_classes, activation='linear')(dense)
        output = Dense(1, activation='tanh')(x)
    model = Model(inputs=inputs, outputs=output)
    return model


def model_mlp_static(data_dim, num_classes, method='category'):
    input = Input(shape=(data_dim,))
    dense = Dense(data_dim * 4, activation='tanh')(input)
    dense = Dense(data_dim * 4, activation='tanh')(dense)
    if method == 'category':
        output = Dense(num_classes, activation='softmax')(dense)
    else:
        x = Dense(num_classes, activation='linear')(dense)
        output = Dense(1, activation='tanh')(x)
    model = Model(inputs=[input], outputs=output)
    return model


class MLP(StrategyBase):
    def __init__(self, train_data, train_label, fit_data,
                 target_filter, train_filter, month, type='static',
                 kernel=None, method='category'):
        super().__init__(train_data, train_label, fit_data,
                         target_filter, train_filter, month,
                         kernel=kernel, method=method)
        self.type = type
        if method == 'category':
            self.train_label = to_categorical(self.train_label)
        self.data_preprocessing()

    def data_preprocessing(self):
        self.train_data = shuffle(self.train_data)
        if self.type == 'static':
            self.train_data = scale(self.train_data[:, 0, :].reshape([len(self.train_data), -1]))
            self.fit_data = scale(self.fit_data[:, 0, :].reshape([len(self.fit_data), -1]))
        else:
            self.train_data = scale(self.train_data.reshape([len(self.train_data), -1])).reshape(
                [len(self.train_data), self.time_steps, -1]).transpose([2, 0, 1])
            self.fit_data = scale(self.fit_data.reshape([len(self.fit_data), -1])).reshape(
                [len(self.fit_data), self.time_steps, -1]).transpose([2, 0, 1])

    def fit_predict(self, class_weight=None):
        print(self.train_data.shape)
        print(self.fit_data.shape)
        result = PowerSetsRecursive(list(range(self.data_dim)))[1:]
        model_metric = []
        for iter in result:
            if self.type == 'static':
                model = model_mlp_static(len(iter), self.num_classes, method=self.method)
                model.compile(loss=self.loss, optimizer='Adam', metrics=['accuracy'])
                model.fit(self.train_data[:, iter], self.train_label, epochs=100, verbose=0,
                          class_weight=class_weight, validation_split=0.1, callbacks=[early_stop])
                metric = model.evaluate(self.train_data[:, iter], self.train_label)
            else:
                model = model_mlp_dynamic(self.time_steps, len(iter), self.num_classes, method=self.method)
                model.compile(loss=self.loss, optimizer='Adam', metrics=['accuracy'])
                model.fit([self.train_data[i] for i in iter], self.train_label, epochs=100, verbose=0,
                          class_weight=class_weight, validation_split=0.1, callbacks=[early_stop])
                metric = model.evaluate([self.train_data[i] for i in iter], self.train_label)
            model_metric.append(metric)
            print(iter, '\t', metric)
            del model
            gc.collect()

        model_metric = np.array(model_metric)
        lest_loss = np.argmin(model_metric[:, 0])
        best_iter = result[lest_loss]
        print('best iter:', best_iter)

        if self.type == 'static':
            self.model = model_mlp_static(len(best_iter), self.num_classes, method=self.method)
            self.fit_data = self.fit_data[:, best_iter]
        else:
            self.model = model_mlp_dynamic(self.time_steps, len(best_iter), self.num_classes, method=self.method)
            self.fit_data = self.fit_data[best_iter].tolist()
        self.predict = self.model.predict(self.fit_data)
        self.predict_transform()
