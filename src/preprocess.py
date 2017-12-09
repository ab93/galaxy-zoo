import config as cfg
import numpy as np
import cv2


def resize(image_name):
    """
    Returns the re-sized image for better ROI.

    :param image_name: Full path of the image

    :return: Image matrix in Numpy

    """

    img = cv2.imread(image_name)
    return cv2.resize(img, None, fx=cfg.IMAGE_SCALE, fy=cfg.IMAGE_SCALE)


def get_histogram_centroid(histogram, bins):
    """
    Return centroid of the histogram

    :param histogram: Histogram array
    :param bins: Bins array

    :return: Centroid of the histogram
    """

    return np.dot(histogram, bins[1:]) / np.sum(histogram)


def get_histogram_entropy(histogram, bins):
    """

    :param histogram: Histogram array
    :param bins: Bins array

    :return: Entropy of the histogram
    """

    return np.dot(histogram, np.log(histogram / np.diff(bins)))


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


def extract_histogram_vectors(img):
    """
    Returns feature vectors consisting of RGB, HSV, YCbCr and CIE Lab color channels.

    :param img: Image matrix

    :return: Vector containing the histogram vectors
    """

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    vector = np.array([])
    vector = np.append(vector, get_color_space_vector(img[:, :, 0], 8, 0, 256))
    vector = np.append(vector, get_color_space_vector(img[:, :, 1], 8, 0, 256))
    vector = np.append(vector, get_color_space_vector(img[:, :, 2], 8, 0, 256))

    vector = np.append(vector, get_color_space_vector(img_hsv[:, :, 0], 8, 0, 1))
    vector = np.append(vector, get_color_space_vector(img_hsv[:, :, 1], 8, 0, 1))
    vector = np.append(vector, get_color_space_vector(img_hsv[:, :, 2], 8, 0, 256))

    return vector

