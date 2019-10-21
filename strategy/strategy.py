from sklearn.preprocessing import scale
import numpy as np
from abc import abstractmethod

low = -0.01
high = 0.01


def to_category(label):
    label = np.where(label < low, -1, label)
    label = np.where(label > high, 1, label)
    label = np.where(abs(label) != 1, 0, label)
    label += 1
    return label


def standardise(data):
    data = np.nan_to_num(data)
    return data


def print_summary(label):
    mean = np.mean(label)
    std = np.std(label)
    print('MARKET SUMMARY:\n', 'mean=\t', mean, '\tstd=\t', std)
    print('min=\t', np.min(label), '\tmax=\t', np.max(label))
    print('left-outlier=\t', np.sum(label < mean - 3 * std), '\tright-outlier=\t', np.sum(label > mean + 3 * std))


class StrategyBase(object):
    def __init__(self, train_data, train_label, fit_data,
                 target_filter, train_filter, month, split=None,
                 kernel=None, method='category', pre_acc=1):
        self.pre_acc = pre_acc
        self.month = month
        self.model = None
        self.predict = None
        self.method = method
        self.kernel = kernel
        self.target_filter = target_filter
        self.train_filter = train_filter
        self.num_classes = 3
        self.fit_data = standardise(fit_data[target_filter])
        self.train_data = standardise(train_data[train_filter])
        self.train_label = np.ravel(train_label)[train_filter]
        # ----------------------------SUMMARY-----------------------
        print_summary(self.train_label)
        print('training size(samples\tdata_dim\ttime_steps):')
        # ----------------------into MULTI-LABEL--------------------
        if self.method == 'category':
            self.train_label = to_category(self.train_label)
            self.loss = 'categorical_crossentropy'
        else:
            self.loss = 'mse'
        self.train_data = self.train_data.swapaxes(1, 2)
        self.fit_data = self.fit_data.swapaxes(1, 2)
        self.samples, self.time_steps, self.data_dim = self.train_data.shape
        print(self.train_data.shape)
        print(self.fit_data.shape)
        if split is not None:
            self.data_dim1 = split + 1
            self.data_dim2 = self.data_dim - self.data_dim1
            self.split_data()

    @abstractmethod
    def get_model(self, class_weight=None):
        '''model construct'''

    @abstractmethod
    def fit_model(self, class_weight=None):
        '''model fit'''

    def get_predict(self):
        self.predict = self.model.predict(self.fit_data)

    def fit_predict(self, class_weight=None):
        if self.month > 1:
            self.load_models()  # NEED OVERLOAD
        else:
            self.get_model(class_weight)  # NEED IMPLEMENT
        self.fit_model(class_weight)  # NEED IMPLEMENT
        self.get_predict()  # NEED OVERLOAD
        self.predict_transform()
        self.save_models()  # NEED OVERLOAD

    def predict_transform(self):
        if self.method == 'category':
            if self.predict.ndim > 1:
                # pre_sum = np.sum(self.predict, axis=1)
                # self.predict = np.matmul(self.predict, np.array([-1, 0, 1]).T) / pre_sum
                self.predict = np.argmax(self.predict, axis=1) - 1
                # self.predict = to_target(self.predict)
            else:
                self.predict -= 1
        self.predict = np.ravel(self.predict)
        predict = np.full(len(self.target_filter), fill_value=-1, dtype=float)
        predict[self.target_filter] = self.predict
        self.predict = predict

    def save_models(self):
        pass

    def load_models(self):
        pass

    def split_data(self):
        self.train_data = [
            self.train_data[:, :, :self.data_dim1],
            scale(
                self.train_data[:, 0, -self.data_dim2:].reshape(
                    [self.train_data.shape[0], -1]
                )
            )
        ]
        self.fit_data = [
            self.fit_data[:, :, :self.data_dim1],
            scale(
                self.fit_data[:, 0, -self.data_dim2:].reshape(
                    [self.fit_data.shape[0], -1]
                )
            )
        ]
