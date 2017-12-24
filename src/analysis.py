from scipy import stats
import numpy as np
import pandas as pd
import config as cfg


def get_target_data(class_id):
    target_column = "Class{}".format(class_id)
    return pd.read_csv(cfg.TRAIN_SOLUTION_FILE, usecols=["GalaxyID", target_column])


def calculate_f_scores():
    target_df = pd.read_csv(cfg.TRAIN_SOLUTION_FILE)
    feature_df = pd.read_csv(cfg.TRAIN_DATA_FILE)

    print target_df['GalaxyID']
    print feature_df['GalaxyID']
    print target_df['GalaxyID'] == feature_df['GalaxyID']


if __name__ == '__main__':
    calculate_f_scores()
