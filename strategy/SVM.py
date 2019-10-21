from strategy import StrategyBase
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.externals import joblib
from sklearn.svm import SVC
from func import get_features

strategy_name = 'SVM'


class SVM(StrategyBase):
    def __init__(self, train_data, train_label, fit_data,
                 target_filter, train_filter, month, split,
                 kernel='linear', method='category'):
        super().__init__(train_data, train_label, fit_data,
                         target_filter, train_filter, month, split=split,
                         kernel=kernel, method=method)
        self.modelM = None
        self.train_label += 1
        self.data_preprocessing()

    def data_preprocessing(self):
        '''
        self.train_data[0] = scale(get_features(self.train_data[0]))
        self.fit_data[0] = scale(get_features(self.fit_data[0]))
        '''
        self.train_data[0] = scale(self.train_data[0].reshape([len(self.train_data[0]), -1]))
        self.fit_data[0] = scale(self.fit_data[0].reshape([len(self.fit_data[0]), -1]))
        self.train_data[1] = scale(self.train_data[1])
        self.fit_data[1] = scale(self.fit_data[1])
        self.train_data[0], self.train_data[1] = \
            shuffle(
                self.train_data[0], self.train_data[1]
            )

    def get_model(self, class_weight=None):
        self.model = SVC(kernel=self.kernel)
        self.modelM = SVC(kernel=self.kernel)

    def fit_model(self, class_weight=None):
        self.model.fit(self.train_data[0], self.train_label)
        self.modelM.fit(self.train_data[1], self.train_label)

    def get_predict(self):
        self.predict = self.model.predict(self.fit_data[0])
        self.predict += self.modelM.predict(self.fit_data[1])
        self.predict /= 2
        self.predict -= 1

    def save_models(self):
        joblib.dump(self.model, strategy_name + '.pkl')
        joblib.dump(self.modelM, strategy_name + 'M.pkl')

    def load_models(self):
        self.model = joblib.load(strategy_name + '.pkl')
        self.modelM = joblib.load(strategy_name + 'M.pkl')
