from strategy import StrategyBase
from keras.layers import *
from keras.models import *
from keras_func import reduce_lr, tbCallBack, early_stop, csv_log
from keras.utils import to_categorical
from sklearn.utils import shuffle
from func import emd_data_transform
from layers import stacked_dense, stacked_lstm, attention_3d_block
from sklearn.preprocessing import scale

from sklearn.svm import OneClassSVM

strategy_name = 'EMD-LSTM'


def model_emd_lstm(time_steps, data_dim1, data_dim2, num_classes, method='category'):
    input1 = Input(shape=(time_steps, data_dim1,))
    input2 = Input(shape=(time_steps, data_dim1,))
    input3 = Input(shape=(data_dim2,))

    lstm1 = attention_3d_block(input1)
    lstm2 = attention_3d_block(input2)
    lstm1 = LSTM(data_dim1 * 8, return_sequences=True)(lstm1)
    lstm2 = LSTM(data_dim1 * 8, return_sequences=True)(lstm2)
    lstm1 = LSTM(data_dim1 * 8, return_sequences=True)(lstm1)
    lstm2 = LSTM(data_dim1 * 8, return_sequences=True)(lstm2)

    lstm1 = LSTM(data_dim1 * 4)(lstm1)
    lstm2 = LSTM(data_dim1 * 4)(lstm2)
    mux1 = Add()([lstm1, lstm2])
    dense1 = BatchNormalization()(mux1)

    '''
    lstm1=Flatten()(lstm1)
    lstm2=Flatten()(lstm2)
    '''
    dense2 = stacked_dense(input3, data_dim2 * 4, 2)

    mux = Concatenate()([dense1, dense2])
    x = stacked_dense(mux, (data_dim1 + data_dim2) * 4, 3)
    if method == 'category':
        output = Dense(num_classes, activation='softmax')(x)
    else:
        output = Dense(1, activation='tanh')(x)
    model = Model(inputs=[input1, input2, input3], outputs=output)
    return model


class EMD_LSTMA(StrategyBase):
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

        self.train_data[0] = emd_data_transform(self.train_data[0])
        self.fit_data[0] = emd_data_transform(self.fit_data[0])

        self.train_data[1] = scale(self.train_data[1])
        self.fit_data[1] = scale(self.fit_data[1])
        self.train_data = [self.train_data[0][0], self.train_data[0][1], self.train_data[1]]
        self.fit_data = [self.fit_data[0][0], self.fit_data[0][1], self.fit_data[1]]
        self.train_data[0], self.train_data[1], self.train_data[2] = \
            shuffle(
                self.train_data[0], self.train_data[1], self.train_data[2]
            )

    def get_model(self, class_weight=None):
        self.model = model_emd_lstm(
            self.time_steps, self.data_dim1, self.data_dim2, self.num_classes, method=self.method
        )
        self.model.compile(loss=self.loss, optimizer='Adam', metrics=['accuracy'])

    def fit_model(self, class_weight=None):
        print('\t\t', self.model.metrics_names)
        print(self.train_data[0].shape)
        self.model.fit(
            self.train_data, self.train_label,
            epochs=100, verbose=0, class_weight=class_weight, validation_split=0.1,
            callbacks=[reduce_lr, early_stop], batch_size=40)
        print('train_rate:\t',
              self.model.evaluate(self.train_data, self.train_label))

    def get_predict(self):
        self.predict = self.model.predict(self.fit_data)

    def save_models(self):
        self.model.save(strategy_name + '.h5')

    def load_models(self):
        self.model = load_model(strategy_name + '.h5')
