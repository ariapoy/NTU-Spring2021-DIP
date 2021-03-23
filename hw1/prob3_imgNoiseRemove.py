from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

import utils

import pdb

# load image
sample6_arr = utils.load_img2npArr("sample6.jpg")
sample7_arr = utils.load_img2npArr("sample7.jpg")
sample5_arr = utils.load_img2npArr("sample5.jpg")

# prob (a)
'''
Solutions 
1. low-pass filtering (for uniform noise)
2. non-linear filtering (for impluse noise)
3. fft filtering [Image denoising by FFT](http://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html)
'''

def filter_low_pass(img_arr, b=1, kernel_size=3, pad_method="edge"):
    M, N = img_arr.shape
    result = np.zeros( (M, N) )
    n_pad = kernel_size // 2
    img_expand_arr = np.pad(img_arr, ((n_pad, n_pad), (n_pad, n_pad)), "edge")
    M_expand, N_expand = img_expand_arr.shape
    kernel_range = kernel_size
    # Generate (b + k - 1)**2, and assign b with cross
    mask = np.ones( (kernel_size, kernel_size)  )
    for i in range(-(kernel_size//2), (kernel_size//2)+1 ):
        mask[kernel_size // 2 + i, kernel_size // 2] = b
    for j in range(-(kernel_size//2), (kernel_size//2)+1 ):
        mask[kernel_size // 2, kernel_size // 2 + j] = b
    mask[kernel_size // 2, kernel_size // 2] = b ** 2
    mask = 1 / ( (b + kernel_size - 1)**2 ) * mask
    # convolution/weighted average
    for i in range(M):
        for j in range(N):
            result_part = img_expand_arr[i: i + kernel_range, j: j + kernel_range] * mask
            result[i, j] = np.sum(result_part)
    result = utils.int_round(result)
    return result

def filter_median(img_arr, kernel_size=3, percent=50, pad_method="edge"):
    M, N = img_arr.shape
    result = np.zeros( (M, N) )
    n_pad = kernel_size // 2
    img_expand_arr = np.pad(img_arr, ((n_pad, n_pad), (n_pad, n_pad)), "edge")
    M_expand, N_expand = img_expand_arr.shape
    kernel_range = kernel_size
    # convolution/weighted average
    for i in range(M):
        for j in range(N):
            result_part = img_expand_arr[i: i + kernel_range, j: j + kernel_range]
            #result[i, j] = np.median(result_part)
            result[i, j] = np.percentile(result_part, percent, interpolation="nearest")
    result = utils.int_round(result)
    return result

def filter_fft(img_arr, Name=None, frac=0.1):
    # config
    keep_fraction = frac
    # transform into frequency domain
    img_freq_arr = fftpack.fft2(img_arr)
    result_freq = img_freq_arr.copy()
    r, c = result_freq.shape
    # filter by keep frac
    result_freq[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    result_freq[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    # plot spectrum graph
    if Name is not None:
        utils.plot_spectrum(img_freq_arr, "{0}_FD_origin".format(Name) )
        utils.plot_spectrum(result_freq, "{0}_FD_filtered".format(Name) )
    result_arr = fftpack.ifft2(result_freq).real
    result = utils.int_round(result_arr)
    return result

# Find best params
'''
utils.plot_param_search(filter_low_pass, "b", range(1, 21), sample5_arr, sample6_arr, "sample6")
utils.plot_param_search(filter_low_pass, "kernel_size", range(3, 21, 2), sample5_arr, sample6_arr, "sample6")

utils.plot_param_search(filter_median, "kernel_size", range(3, 21, 2), sample5_arr, sample7_arr, "sample7")
utils.plot_param_search(filter_median, "percent", range(25, 75+1, 5), sample5_arr, sample7_arr, "sample7")

utils.plot_param_search(filter_fft, "frac", [round(0.05 * i, 2) for i in range(1, 10)], sample5_arr, sample6_arr, "sample6")
utils.plot_param_search(filter_fft, "frac", [round(0.05 * i, 2) for i in range(1, 10)], sample5_arr, sample7_arr, "sample7")
'''

# Remember delete it before submit hw1

result8_1_arr = filter_low_pass(sample6_arr, b=2, kernel_size=5)
utils.save_npArr2JPG(result8_1_arr, "8_result")
#result8_2_arr = filter_low_pass(sample6_arr, b=3, kernel_size=5)
#utils.save_npArr2JPG(result8_2_arr, "8_result_low_pass2")

#result8_2_arr = filter_median(sample6_arr, kernel_size=3)
#utils.save_npArr2JPG(result8_2_arr, "8_result_median")

#result9_1_arr = filter_low_pass(sample7_arr, b=1, kernel_size=5)
#utils.save_npArr2JPG(result9_1_arr, "9_result_low_pass")

result9_2_arr = filter_median(sample7_arr, kernel_size=3, percent=50)
utils.save_npArr2JPG(result9_2_arr, "9_result")

#result8_3_arr = filter_fft(sample6_arr, "sample6")
#utils.save_npArr2JPG(result8_3_arr, "8_result_fft")

#result9_3_arr = filter_fft(sample7_arr, "sample7")
#utils.save_npArr2JPG(result9_3_arr, "9_result_fft")

# prob (b)
"""
Question 2
Can I use `np.prod`?
"""
def PSNR(F_true, F):
    MSE = np.sum((F_true - F)**2) / np.prod(F.shape)
    res = 10 * np.log10(255**2 / MSE)
    return res

print('PSNR of random noise by low-pass : {0}'.format(PSNR(sample5_arr, result8_1_arr)))
#print('PSNR of random noise by low-pass : {0}'.format(PSNR(sample5_arr, result8_2_arr)))
#print('PSNR of random noise by fft : {0}'.format(PSNR(sample5_arr, result8_3_arr)))
#print('PSNR of impulse noise by low-pass: {0}'.format(PSNR(sample5_arr, result9_1_arr)))
print('PSNR of impulse noise by median   : {0}'.format(PSNR(sample5_arr, result9_2_arr)))
#print('PSNR of impulse noise by fft: {0}'.format(PSNR(sample5_arr, result9_3_arr)))

