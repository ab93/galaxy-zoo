import os
import pandas as pd
import config as cfg
import csv
from features import Features


def get_galaxy_ids():
    df = pd.read_csv(cfg.TRAIN_SOLUTION_FILE, usecols=[0])
    return df.GalaxyID


def read_images():
    galaxy_ids = get_galaxy_ids()
    for galaxy_id in galaxy_ids:
        image_file = '.'.join([str(galaxy_id), 'jpg'])
        feature_vector = [galaxy_id]
        features = Features(os.path.join(cfg.TRAIN_IMAGE_DIR, image_file))
        feature_vector.extend(features.get_feature_vector())
        yield feature_vector


def write_features():
    with open(cfg.TRAIN_DATA_FILE, 'wb') as train_file:
        csv_writer = csv.writer(train_file)
        count = 0
        csv_writer.writerow(Features.get_header())
        for row in read_images():
            csv_writer.writerow(row)
            print count
            count += 1


if __name__ == '__main__':
    write_features()
