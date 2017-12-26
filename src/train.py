import numpy as np
import pandas as pd
import config as cfg
from sklearn.linear_model import ridge_regression


class TrainModel(object):

    def __init__(self, target):
        self.target = target
        self.x_train = None
        self.y_train = None

    def set_data(self):
        self.x_train = pd.read_csv(cfg.TRAIN_DATA_FILE).drop(['GalaxyID'], axis=1)
        self.y_train = pd.read_csv(cfg.TRAIN_SOLUTION_FILE, usecols=[self.target])

    def grid_search(self):
        pass

    def plot_learning_curve(self):
        pass

    def train(self):
        pass
