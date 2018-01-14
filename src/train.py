from collections import defaultdict
import copy
import numpy as np
import pandas as pd
import config as cfg
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import mean_squared_error, make_scorer, r2_score


class TrainModel(object):

    def __init__(self, target):
        self.target = "".join(['Class', target])
        self.X_train = None
        self.y_train = None

    def set_data(self):
        self.X_train = pd.read_csv(cfg.TRAIN_DATA_FILE).drop(['GalaxyID'], axis=1)
        self.y_train = pd.read_csv(cfg.TRAIN_SOLUTION_FILE, usecols=[self.target])

    def normalize(self):
        pass

    def mean_decrease_score(self):
        rf = RandomForestRegressor()
        scores = defaultdict(list)

        X = copy.deepcopy(self.X_train.values)
        y = copy.deepcopy(self.y_train.values)
        features = self.X_train.columns

        shuffle_split = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
        count = 0

        for train_idx, test_idx in shuffle_split.split(X, y):
            count += 1
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            print "Fitting shuffle splits: {}".format(count)
            rf.fit(X_train, y_train)
            overall_r2_score = r2_score(y_test, rf.predict(X_test))

            for feature_idx in xrange(X.shape[1]):
                X_test_shuffled = X_test.copy()
                np.random.shuffle(X_test_shuffled[:, feature_idx])
                shuffled_r2_score = r2_score(y_test, rf.predict(X_test_shuffled))
                scores[features[feature_idx]].append((overall_r2_score - shuffled_r2_score) / overall_r2_score)

        scores = sorted([(np.mean(score), feature) for feature, score in scores.items()], reverse=True)
        print "Features sorted by their score: \n{}".format(scores)

    def grid_search(self, cv=3):
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        alphas = np.logspace(-2, 3, 5)
        reg = GridSearchCV(Ridge(), param_grid={'alpha': alphas}, scoring=scorer, cv=cv, n_jobs=-1)
        reg.fit(self.X_train, self.y_train)
        print np.sqrt(np.negative(reg.best_score_))

    def plot_learning_curve(self):
        pass

    def train(self):
        pass


if __name__ == '__main__':
    trn = TrainModel('1.1')
    trn.set_data()
    print trn.X_train.shape
    print trn.y_train.shape
    trn.mean_decrease_score()
    # trn.grid_search()
