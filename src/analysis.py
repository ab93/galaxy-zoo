import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import f_regression, mutual_info_regression
import pandas as pd
import config as cfg


def get_target_data(class_id):
    target_column = 'Class{}'.format(class_id)
    return pd.read_csv(cfg.TRAIN_SOLUTION_FILE, usecols=['GalaxyID', target_column])


def calculate_f_scores():
    target_df = pd.read_csv(cfg.TRAIN_SOLUTION_FILE)
    feature_df = pd.read_csv(cfg.TRAIN_DATA_FILE).drop(['GalaxyID'], axis=1)

    for class_name in cfg.TARGET_CLASSES:
        f_score, p_score = f_regression(feature_df, target_df[class_name])
        f_score /= np.max(f_score)
        scores_df = pd.DataFrame({'f_score': f_score, 'p_score': p_score}, index=feature_df.columns)
        scores_df.to_csv(os.path.join(cfg.ANALYSIS_DIR, '{}_scores.csv'.format(class_name)))


def scatter_plot():
    target_df = pd.read_csv(cfg.TRAIN_SOLUTION_FILE)
    feature_df = pd.read_csv(cfg.TRAIN_DATA_FILE).drop(['GalaxyID'], axis=1)

    plt.scatter(feature_df['G_min'], target_df['Class1.1'], edgecolors='black', s=18)
    plt.show()


if __name__ == '__main__':
    # calculate_f_scores()
    scatter_plot()
