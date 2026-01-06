# image_utils.py
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(path):
    """
    Loads an image from the given path and returns it as a numpy array.
    """
    image = Image.open(path)
    return np.array(image)

def edge_detection(image_array):
    """
    Detects edges in the image using Sobel filters.
    Returns the magnitude of the edges.
    """
    # 1. Convert to grayscale (mean of RGB)
    gray_image = np.mean(image_array, axis=2)

    # 2. Define Sobel filters
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # 3. Apply convolution
    edge_x = convolve2d(gray_image, filter_x, mode='same', boundary='fill', fillvalue=0)
    edge_y = convolve2d(gray_image, filter_y, mode='same', boundary='fill', fillvalue=0)

    # 4. Calculate magnitude
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    return edge_mag
