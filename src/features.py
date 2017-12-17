import config as cfg
import numpy as np
import cv2


class Features(object):

    num_bins = 8

    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        self.feature_vector = None
        self._resize()
        self.create_feature_vector()

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

        hist, bins = np.histogram(channel, num_bins, (min_value, max_value), density=True)
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
            vector.extend(self.get_color_space_vector(self.img[:, :, idx], self.num_bins, 0, 256))
        return vector

    def extract_hsv_histogram(self):
        """
        Returns feature vectors consisting of HSV color channels.

        :return: List containing histogram features
        """

        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        vector = []

        vector.extend(self.get_color_space_vector(img_hsv[:, :, 0], self.num_bins, 0, 180))
        vector.extend(self.get_color_space_vector(img_hsv[:, :, 1], self.num_bins, 0, 256))
        vector.extend(self.get_color_space_vector(img_hsv[:, :, 2], self.num_bins, 0, 256))

        return vector

    def get_feature_vector(self):
        return self.feature_vector

    def create_feature_vector(self):
        # Append RGB features
        feature_vector = self.extract_rgb_histogram()

        # Append HSV features
        feature_vector.extend(self.extract_hsv_histogram())

        self.feature_vector = np.asarray(feature_vector)

    @classmethod
    def get_header(cls):
        stats = [str(bin_) for bin_ in range(cls.num_bins)]
        stats.extend(['max', 'min', 'median'])

        bgr_color_codes = ['B', 'G', 'R']
        bgr_header = ["{}_{}".format(channel, stat) for channel in bgr_color_codes for stat in stats]

        hsv_color_codes = ['H', 'S', 'V']
        hsv_header = ["{}_{}".format(channel, stat) for channel in hsv_color_codes for stat in stats]

        return ['galaxy_id'] + bgr_header + hsv_header
