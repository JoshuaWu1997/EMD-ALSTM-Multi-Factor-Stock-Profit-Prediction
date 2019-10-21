from strategy import StrategyBase
from keras.layers import *
from keras.models import *
from keras.utils import to_categorical
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from layers import stacked_dense
from keras_func import reduce_lr, early_stop, csv_log

strategy_name = 'MLP'


def model_mlp(time_steps, data_dim1, data_dim2, num_classes, method='category'):
    inputs = []
    denses = []
    for i in range(data_dim1):
        input = Input(shape=(time_steps,))
        dense = Dense(time_steps, activation='relu')(input)
        dense_norm = BatchNormalization()(dense)
        inputs.append(input)
        denses.append(dense_norm)
    input = Input(shape=(data_dim2,))
    dense = Dense(data_dim2, activation='tanh')(input)
    dense_norm = BatchNormalization()(dense)
    denses.append(dense_norm)
    inputs.append(input)

    mux1 = Add()(denses[:-1])
    mux = Concatenate()([mux1, denses[-1]])
    dense = Dense((time_steps + data_dim2) * 4, activation='tanh')(mux)
    dense = Dense((time_steps + data_dim2) * 4, activation='tanh')(dense)

    if method == 'category':
        output = Dense(num_classes, activation='softmax')(dense)
    else:
        x = Dense(num_classes, activation='linear')(dense)
        output = Dense(1, activation='tanh')(x)
    model = Model(inputs=inputs, outputs=output)
    return model


class MLP(StrategyBase):
    def __init__(self, train_data, train_label, fit_data,
                 target_filter, train_filter, month, split,
                 kernel=None, method='category'):
        super().__init__(train_data, train_label, fit_data,
                         target_filter, train_filter, month, split=split,
                         kernel=kernel, method=method)
        if method == 'category':
            self.train_label = to_categorical(self.train_label)
        self.data_preprocessing()

    def data_preprocessing(self):
        self.train_data[0] = scale(self.train_data[0].reshape([len(self.train_data[0]), -1])).reshape(
            [len(self.train_data[0]), self.time_steps, self.data_dim1])
        self.fit_data[0] = scale(self.fit_data[0].reshape([len(self.fit_data[0]), -1])).reshape(
            [len(self.fit_data[0]), self.time_steps, self.data_dim1])
        self.train_data[1] = scale(self.train_data[1])
        self.fit_data[1] = scale(self.fit_data[1])
        self.train_data[0], self.train_data[1] = \
            shuffle(
                self.train_data[0], self.train_data[1]
            )
        self.train_data[0] = self.train_data[0].transpose([2, 0, 1])
        self.fit_data[0] = self.fit_data[0].transpose([2, 0, 1])
        train_data = []
        fit_data = []
        for i in range(self.data_dim1):
            train_data.append(self.train_data[0][i])
            fit_data.append(self.fit_data[0][i])
        train_data.append(self.train_data[1])
        fit_data.append(self.fit_data[1])
        self.train_data = train_data
        self.fit_data = fit_data

    def get_model(self, class_weight=None):
        self.model = model_mlp(self.time_steps, self.data_dim1, self.data_dim2, self.num_classes, method=self.method)
        self.model.compile(loss=self.loss, optimizer='Adam', metrics=['accuracy'])

    def fit_model(self, class_weight=None):
        print('\t\t', self.model.metrics_names)
        self.model.fit(
            self.train_data, self.train_label, epochs=500, verbose=0,
            class_weight=class_weight,
            callbacks=[reduce_lr, csv_log, early_stop])
        print('train_rate:\t',
              self.model.evaluate(self.train_data, self.train_label))

    def get_predict(self):
        self.predict = self.model.predict(self.fit_data)

    def save_models(self):
        self.model.save(strategy_name + '.h5')

    def load_models(self):
        self.model = load_model(strategy_name + '.h5')
