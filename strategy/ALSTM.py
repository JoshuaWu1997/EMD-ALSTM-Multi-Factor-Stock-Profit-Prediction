from strategy import StrategyBase
from keras.layers import *
from keras.models import *
from keras_func import reduce_lr, tbCallBack, early_stop, csv_log
from keras.utils import to_categorical
from sklearn.utils import shuffle
from layers import stacked_dense, stacked_lstm, attention_3d_block
from sklearn.preprocessing import scale


strategy_name = 'Attention-LSTM'


def model_lstm(time_steps, data_dim1, data_dim2, num_classes, method='category'):
    input1 = Input(shape=(time_steps, data_dim1,))
    input2 = Input(shape=(data_dim2,))

    lstm1 = attention_3d_block(input1)
    lstm1 = LSTM(data_dim1 * 8, return_sequences=True)(lstm1)
    lstm1 = LSTM(data_dim1 * 8, return_sequences=True)(lstm1)
    lstm1 = LSTM(data_dim1 * 4)(lstm1)

    dense1 = BatchNormalization()(lstm1)
    dense2 = stacked_dense(input2, data_dim2 * 4, 2)

    mux = Concatenate()([dense1, dense2])
    x = stacked_dense(mux, (data_dim1 + data_dim2) * 4, 4)
    if method == 'category':
        output = Dense(num_classes, activation='softmax')(x)
    else:
        output = Dense(1, activation='linear')(x)
    model = Model(inputs=[input1, input2], outputs=output)
    return model


def model_lstm_flatten(time_steps, data_dim1, data_dim2, num_classes, method='category'):
    input1 = Input(shape=(time_steps, data_dim1,))
    input2 = Input(shape=(data_dim2,))

    lstm1 = attention_3d_block(input1)
    lstm1 = LSTM(data_dim1 * 8, return_sequences=True)(lstm1)
    lstm1 = LSTM(data_dim1 * 8, return_sequences=True)(lstm1)
    lstm1 = Flatten()(lstm1)
    dense1 = BatchNormalization()(lstm1)

    mux = Concatenate()([dense1, input2])
    x = stacked_dense(mux, (data_dim1 * time_steps + data_dim2) * 4, 2)
    x = stacked_dense(mux, (data_dim1 * time_steps + data_dim2) * 4, 2)
    if method == 'category':
        output = Dense(num_classes, activation='softmax')(x)
    else:
        output = Dense(1, activation='linear')(x)
    model = Model(inputs=[input1, input2], outputs=output)
    return model


class LSTMA(StrategyBase):
    def __init__(self, train_data, train_label, fit_data,
                 target_filter, train_filter, month, split,
                 kernel=None, method='category', pre_acc=1):
        super().__init__(train_data, train_label, fit_data,
                         target_filter, train_filter, month, split=split,
                         kernel=kernel, method=method, pre_acc=pre_acc)
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
        self.train_data[0], self.train_data[1] = shuffle(
            self.train_data[0], self.train_data[1]
        )

    def get_model(self, class_weight=None):
        self.model = model_lstm(
            self.time_steps, self.data_dim1, self.data_dim2, self.num_classes, method=self.method
        )
        self.model.compile(loss=self.loss, optimizer='Adam', metrics=['accuracy'])

    def fit_model(self, class_weight=None):
        print('\t\t', self.model.metrics_names)
        self.model.fit(
            self.train_data, self.train_label,
            epochs=200, verbose=0, validation_split=0.1, batch_size=50,
            class_weight=class_weight,
            callbacks=[reduce_lr, early_stop, csv_log])
        print('train_rate:\t',
              self.model.evaluate(self.train_data, self.train_label))

    def get_predict(self):
        self.predict = self.model.predict(self.fit_data)

    def save_models(self):
        self.model.save(strategy_name + '.h5')

    def load_models(self):
        self.model = load_model(strategy_name + '.h5')
