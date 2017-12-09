import config as cfg
import numpy as np
from skimage.feature import greycomatrix
import cv2


class Features(object):

    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        self.vector = None
        self._resize()

    def _resize(self):
        self.img = cv2.resize(self.img, None, fx=cfg.IMAGE_SCALE, fy=cfg.IMAGE_SCALE)

    @staticmethod
    def get_histogram_centroid(histogram, bins):
        """
        Return centroid of the histogram

        :param histogram: Histogram array
        :param bins: Bins array

        :return: Centroid of the histogram
        """

        return np.dot(histogram, bins[1:]) / np.sum(histogram)

    @staticmethod
    def get_histogram_entropy(histogram, bins):
        """

        :param histogram: Histogram array
        :param bins: Bins array

        :return: Entropy of the histogram
        """

        return np.dot(histogram, np.log(histogram / np.diff(bins)))

    @staticmethod
    def get_color_space_vector(channel, num_bins, min_value, max_value):
        """
        Returns the color space vector of a channel

        :param channel: Numpy array containing the channel
        :param num_bins: Number of bins
        :param min_value: Minimum value
        :param max_value: Maximum value

        :return: Numpy vector containing the histogram

        """

        hist, bins = np.histogram(channel, num_bins, min_value, max_value, density=True)
        max_hist_value = np.max(hist)
        min_hist_value = np.min(hist)
        median = np.median(hist)
        return np.concatenate((hist, [max_hist_value, min_hist_value, median]))

    def extract_rgb_histogram(self):
        """
        Returns feature vectors consisting of RGB color channels.

        :return: List containing histogram features
        """

        vector = []
        for idx in range(3):
            vector.extend(self.get_color_space_vector(self.img[:, :, idx], 8, 0, 256))
        return vector

    def extract_hsv_histogram(self):
        """
        Returns feature vectors consisting of HSV color channels.

        :return: List containing histogram features
        """

        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        vector = []

        vector.extend(self.get_color_space_vector(img_hsv[:, :, 0], 8, 0, 180))
        vector.extend(self.get_color_space_vector(img_hsv[:, :, 1], 8, 0, 256))
        vector.extend(self.get_color_space_vector(img_hsv[:, :, 2], 8, 0, 256))

        return vector

    def get_feature_vector(self):
        pass


