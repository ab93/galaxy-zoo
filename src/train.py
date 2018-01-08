import numpy as np
import pandas as pd
import config as cfg
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer


class TrainModel(object):

    def __init__(self, target):
        self.target = "".join(['Class', target])
        self.x_train = None
        self.y_train = None

    def set_data(self):
        self.x_train = pd.read_csv(cfg.TRAIN_DATA_FILE).drop(['GalaxyID'], axis=1)
        self.y_train = pd.read_csv(cfg.TRAIN_SOLUTION_FILE, usecols=[self.target])

    def normalize(self):
        pass

    def grid_search(self, cv=3):
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        alphas = np.logspace(-2, 3, 5)
        reg = GridSearchCV(Ridge(), param_grid={'alpha': alphas}, scoring=scorer, cv=cv, n_jobs=-1)
        reg.fit(self.x_train, self.y_train)
        print np.sqrt(np.negative(reg.best_score_))

    def plot_learning_curve(self):
        pass

    def train(self):
        pass


if __name__ == '__main__':
    trn = TrainModel('1.1')
    trn.set_data()
    print trn.x_train.shape
    print trn.y_train.shape
    trn.grid_search()
