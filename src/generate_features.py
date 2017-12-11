import os
import numpy as np
import pandas as pd
import config as cfg
import csv
from preprocess import Features


def write_features(image_name, feature_row):
    galaxy_id = image_name.split('.')[0]
    row = pd.Series(np.concatenate(([galaxy_id], feature_row)))
    with open(os.path.join(cfg.TRAIN_DATA_FILE, 'training_data.csv')) as train_file:
        csv_writer = csv.writer(train_file)
        csv_writer.writerow(row)



def read_images():

    for image_file in os.listdir(cfg.TRAIN_IMAGE_DIR):
        features = Features(os.path.join(cfg.TRAIN_IMAGE_DIR, image_file))
        feature_vector = features.get_feature_vector()
        write_features(image_file, feature_vector)


if __name__ == '__main__':
    read_images()
