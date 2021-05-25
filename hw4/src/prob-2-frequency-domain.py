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

def filter_on_freqdomain(img_arr, filter_name="Gaussian_low-pass", D0=10, D1=1000):
    """
    D0: cutoff freq of Gaussian filter (\sigma)
    """
    F = shift_fft(img_arr)
    M, N = img_arr.shape
    if filter_name == "Gaussian_low-pass":
        D = np.fromfunction(lambda u, v: np.sqrt( (u - M//2)**2 + (v - N//2)**2), (M, N))
        H = np.exp(-D**2/(2*D0**2))
    elif filter_name == "Gaussian_high-pass":
        D = np.fromfunction(lambda u, v: np.sqrt( (u - M//2)**2 + (v - N//2)**2), (M, N))
        H = 1 - np.exp(-D**2/(2*D0**2))
        H = np.exp(-D**2/(2*D1**2)) - np.exp(-D**2/(2*D0**2))
    elif filter_name == "Laplacian_high-pass":
        D = np.fromfunction(lambda u, v: np.sqrt( (u - M//2)**2 + (v - N//2)**2), (M, N))
        H = -(D)
    G = H*F
    return G

def ifft_shift(img_arr):
    def shift_arr(img_arr):
        shift_scaler = np.fromfunction(lambda i, j: (-1)**(i + j), img_arr.shape)
        res_arr = img_arr*shift_scaler
        return res_arr
    img_center = ifft2(img_arr)
    g = shift_arr(img_center)
    res_arr = utils.int_round(g)
    return res_arr

def filter_on_spatdomain(img_arr, filter_name="Gaussian_low-pass", D0=10, D1=1000, k=5):
    """
    D0: cutoff freq of Gaussian filter (\sigma)
    k: kernel size of filter
    """
    M, N = img_arr.shape
    if filter_name == "Gaussian_low-pass":
        h = np.fromfunction(lambda u, v: 1/(2*np.pi*D0**2)*np.exp(-((u - k//2)**2 + (v - k//2)**2)/(2*D0**2)), (k, k) )
        #h = np.fromfunction(lambda u, v: np.exp(-((u - k//2)**2 + (v - k//2)**2)/(2*D0**2)), (k, k) )
        h = h / np.sum(h)
    elif filter_name == "Gaussian_high-pass":
        h2 = np.fromfunction(lambda u, v: 1/(2*np.pi*D0**2)*np.exp(-((u - k//2)**2 + (v - k//2)**2)/(2*D0**2)), (k, k) )
        h1 = np.fromfunction(lambda u, v: 1/(2*np.pi*D1**2)*np.exp(-((u - k//2)**2 + (v - k//2)**2)/(2*D1**2)), (k, k) )
        h = h2 - h1
        #h = np.fromfunction(lambda u, v: 1/(2*np.pi*D0**2)*np.exp(-((u - k//2)**2 + (v - k//2)**2)/(2*D0**2)), (k, k) )
        h = -1/h[0, 0]*h
        h[k//2, k//2] = -(h.sum() - h[k//2,k//2]) + h[k//2,k//2]
    elif filter_name == "Laplacian_high-pass":
        h = np.array([
                      [0,  1, 0], 
                      [1, -4, 1],
                      [0,  1, 0]
                     ])

    res_arr = convolve2d(img_arr, h, boundary="symm", mode="same")
    return res_arr

result6_arr = ifft_shift(filter_on_freqdomain(sample2_arr, D0=50) )
utils.save_npArr2JPG(result6_arr, "result6")
result7_arr = filter_on_spatdomain(sample2_arr, D0=50, k=5)
utils.save_npArr2JPG(result7_arr, "result7")
print("Finish prob 2 (b).")

# prob 2(c)
"""
Based on the result of part (a), design and apply a high-pass filter in the frequency domain and transform the result back to the pixel domain by inverse Fourier transform. The resultant image is saved as \textbf{result8.png}. Please also design a high-pass filter in the pixel domain which behaves similarly to the one you design in the frequency domain. Output the result as \textbf{result9.png} and provide some discussions.
"""

result8_G_arr = ifft_shift(filter_on_freqdomain(sample2_arr, filter_name="Gaussian_high-pass", D0=50) )
utils.save_npArr2JPG(result8_G_arr, "tmp/result8-Gaussian")
result9_G_arr = filter_on_spatdomain(sample2_arr, filter_name="Gaussian_high-pass", D0=50, k=5)
utils.save_npArr2JPG(result9_G_arr, "tmp/result9-Gaussian")
result8_arr = ifft_shift(filter_on_freqdomain(sample2_arr, filter_name="Laplacian_high-pass", D0=50) )
utils.save_npArr2JPG(result8_arr, "result8")
result9_arr = filter_on_spatdomain(sample2_arr, filter_name="Laplacian_high-pass", D0=50, k=5)
utils.save_npArr2JPG(result9_G_arr, "result9")
print("Finish prob 2 (c).")

