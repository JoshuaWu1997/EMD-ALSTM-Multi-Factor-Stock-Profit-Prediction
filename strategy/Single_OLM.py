from strategy import StrategyBase
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import KernelPCA
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

strategy_name = 'OLM'


class OLM(StrategyBase):
    def __init__(self, train_data, train_label, fit_data,
                 target_filter, train_filter, month,
                 kernel=None, method='regression'):
        super().__init__(train_data, train_label, fit_data,
                         target_filter, train_filter, month,
                         kernel=kernel, method=method)
        self.mse = None
        self.r2 = None
        self.data_preprocessing()
        if self.kernel is not None:
            pca = KernelPCA(n_components=5, kernel=self.kernel)
            pca.fit(self.train_data)
            self.train_data = pca.transform(self.train_data)
            self.fit_data = pca.transform(self.fit_data)
            print('After PCA_' + self.kernel + ':\n', pca.lambdas_ / np.sum(pca.lambdas_))

    def get_mse(self):
        pre = self.model.predict(self.train_data)
        self.mse = mean_squared_error(self.train_label, np.ravel(pre))
        self.r2 = r2_score(self.train_label, np.ravel(pre))
        print('mse:\t', self.mse, 'Rsqrt:\t', self.r2)

    def data_preprocessing(self):
        '''
        self.train_data = self.train_data[:, 0, -1]
        self.fit_data = self.fit_data[:, 0, -1]
        '''
        self.train_data = self.train_data.reshape([len(self.train_data), -1])
        self.fit_data = self.fit_data.reshape([len(self.fit_data), -1])
        self.train_data = preprocessing.scale(self.train_data)
        self.fit_data = preprocessing.scale(self.fit_data)

    def get_model(self, class_weight=None):
        self.model = LinearRegression()

    def fit_model(self, class_weight=None):
        self.model.fit(self.train_data, self.train_label)
        self.get_mse()
        print('coef:\n', np.ravel(self.model.coef_), self.model.intercept_)
        print('train_rate:\t', self.model.score(self.train_data, self.train_label))

    def save_models(self):
        joblib.dump(self.model, strategy_name + '.pkl')

    def load_models(self):
        self.model = joblib.load(strategy_name + '.pkl')
