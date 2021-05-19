from PIL import Image
import numpy as np
from numpy.linalg import matrix_power
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

import utils

import pdb

import warnings
warnings.filterwarnings("ignore")

sample1_arr = utils.load_img2npArr("sample1.png")
print(sample1_arr.shape)

# prob 1(a)
"""
Perform dithering using the dither matrix \(I_{2}\) in Figure 1.(b) and output the result as \textbf{result1.png}
"""

def init_dither_mat(I, dim=2):
    return I

def dithering(img_arr, I, noise_type="white"):
    """
    step 1. add noise, white, pink, blue noise in Lec 6 page 13
    step 2. threshold matrix: 255*(I + 0.5)/(I.shape[0])**2
    step 3. dithering by threshold matrix
    """
    # ref: [Generate colors of noise in Python](https://stackoverflow.com/questions/67085963/generate-colors-of-noise-in-python)
    def noise_psd(N, psd = lambda f: 1):
        X_white = np.fft.rfft(np.random.randn(N))
        S = psd(np.fft.rfftfreq(N))
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S
        return np.fft.irfft(X_shaped)
    def PSDGenerator(f):
        return lambda N: noise_psd(N, f)
    @PSDGenerator
    def white_noise(f):
        return 1
    @PSDGenerator
    def blue_noise(f):
        return np.sqrt(f)
    @PSDGenerator
    def pink_noise(f):
        return 1/np.where(f == 0, float('inf'), np.sqrt(f))

    return img_arr

dither_mat = init_dither_mat(np.array([[1, 2], 
                                       [3, 0]])
                             )


