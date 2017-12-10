import os
import numpy as np
import pandas as pd
import config as cfg
from preprocess import Features


def write_features(feature_row):
    pass

def read_images():
    for image_file in os.listdir(cfg.TRAIN_IMAGE_DIR):
        features = Features(os.path.join(cfg.TRAIN_IMAGE_DIR, image_file))
        feature_vector = features.get_feature_vector()
        write_features(feature_vector)


if __name__ == '__main__':
    read_images()
