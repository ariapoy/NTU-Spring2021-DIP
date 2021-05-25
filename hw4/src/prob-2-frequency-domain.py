from PIL import Image, ImageDraw
import numpy as np
from numpy.linalg import matrix_power
from scipy.signal import convolve2d
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

import utils

import pdb

import warnings
warnings.filterwarnings("ignore")

# reproducible
SEED = 0
np.random.seed(SEED)

sample2_arr = utils.load_img2npArr("sample2.png")
sample2_arr = sample2_arr[:, :, 0]

# prob 2(a)
"""
Perform Fourier transform on \textbf{sample2.png} to obtain its frequency spectrum and output it as \textbf{result5.png}. (Please take the log magnitude of the absolute value and center the low frequency part at the origin for visualization.)
"""

def shift_fft(img_arr):
    def shift_arr(img_arr):
        shift_scaler = np.fromfunction(lambda i, j: (-1)**(i + j), img_arr.shape)
        res_arr = img_arr*shift_scaler
        return res_arr
    img_shift = shift_arr(img_arr)
    img_center_fft = fft2(img_shift)
    return img_center_fft

def plot_fft(img_arr):
    img_fft_abslog = np.log( np.abs(img_arr) )
    img_fft_abslog_max, img_fft_abslog_min = img_fft_abslog.max(), img_fft_abslog.min()
    # scale range to [0, 1]
    img_fft_plot = (img_fft_abslog - img_fft_abslog_min) / (img_fft_abslog_max - img_fft_abslog_min)
    return img_fft_plot

result5_arr = utils.int_round(255*plot_fft(shift_fft(sample2_arr) ) )
utils.save_npArr2JPG(result5_arr, "result5")
print("Finish prob 2 (a).")

# prob 2(b)
"""
Based on the result of part (a), design and apply a low-pass filter in the frequency domain and transform the result back to the pixel domain by inverse Fourier transform. The resultant image is saved as \textbf{result6.png}. Please also design a low-pass filter in the pixel domain which behaves similarly to the one you design in the frequency domain. Output the result as \textbf{result7.png} and provide some discussions.
"""

def filter_on_freqdomain(img_arr, filter_name="Gaussian_low-pass"):
    F = shift_fft(img_arr)
    if filter_name == "Gaussian_low-pass":
        D0 = 10
        H = None
    pdb.set_trace()
    return img_arr

result6_arr = filter_on_freqdomain(sample2_arr)

