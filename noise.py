import numpy as np


def gauss(image, mean=0.0, var=0.1):
    sigma = var ** 0.5
    g = np.random.normal(mean, sigma, image.shape)
    noisy = image + g
    return noisy


def test_noise(img, noisy, labels):
    pass