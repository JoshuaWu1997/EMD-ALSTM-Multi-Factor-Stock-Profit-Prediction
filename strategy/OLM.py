from strategy import StrategyBase
import numpy as np
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import KernelPCA
from sklearn.externals import joblib
from sklearn import preprocessing
from func import get_features

strategy_name = 'OLM'


class OLM(StrategyBase):
    def __init__(self, train_data, train_label, fit_data,
                 target_filter, train_filter, month, split,
                 kernel=None, method='regression'):
        super().__init__(train_data, train_label, fit_data,
                         target_filter, train_filter, month, split=split,
                         kernel=kernel, method=method)
        self.modelM = None
        self.data_preprocessing()
        if self.kernel is not None:
            pca = KernelPCA(n_components=5, kernel=self.kernel)
            pca.fit(self.train_data[0])
            self.train_data[0]= pca.transform(self.train_data[0])
            self.fit_data[0] = pca.transform(self.fit_data[0])
            print('After PCA_' + self.kernel + ':\n', pca.lambdas_ / np.sum(pca.lambdas_))

    def data_preprocessing(self):
        self.train_data[0] = scale(self.train_data[0].reshape([len(self.train_data[0]), -1]))
        self.fit_data[0] = scale(self.fit_data[0].reshape([len(self.fit_data[0]), -1]))
        self.train_data[1] = scale(self.train_data[1])
        self.fit_data[1] = scale(self.fit_data[1])

    def get_predict(self):
        self.predict = self.model.predict(self.fit_data[0])
        self.predict += self.modelM.predict(self.fit_data[1])
        self.predict /= 2

    def get_model(self, class_weight=None):
        self.model = LinearRegression()
        self.modelM = LinearRegression()

    def fit_model(self, class_weight=None):
        self.model.fit(self.train_data[0], self.train_label)
        print('train_rate:\t', self.model.score(self.train_data[0], self.train_label))
        self.modelM.fit(self.train_data[1], self.train_label)
        print('train_rate:\t', self.modelM.score(self.train_data[1], self.train_label))

    def save_models(self):
        joblib.dump(self.model, strategy_name + '.pkl')
        joblib.dump(self.modelM, strategy_name + 'M.pkl')

    def load_models(self):
        self.model = joblib.load(strategy_name + '.pkl')
        self.modelM = joblib.load(strategy_name + 'M.pkl')
