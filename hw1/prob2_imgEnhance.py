from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

import utils
import pdb
import warnings
warnings.filterwarnings("ignore")

# load image
sample2_arr = utils.load_img2npArr("sample2.jpg")
sample3_arr = utils.load_img2npArr("sample3.jpg")
sample4_arr = utils.load_img2npArr("sample4.jpg")

# prob (a)
result3_arr = utils.int_round(sample2_arr / 5)
utils.save_npArr2JPG(result3_arr, "3_result")

# prob (b)
result4_arr = utils.int_round(result3_arr * 5)
utils.save_npArr2JPG(result4_arr, "4_result")

# prob (c)
def poy_histogram(img_arr, bins):
    intensity = bins
    cnt = np.zeros(bins.shape)
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            cnt[ img_arr[i, j] ] += 1
    return cnt[:-1], intensity

def intensity_hist(img_arr, n_bins=256):
    bins = np.arange(n_bins + 1)
    cnt, intensity = poy_histogram(img_arr, bins=bins)
    return cnt, intensity[:-1]

L = 256
sample2_cnt, sample2_intensity = intensity_hist(sample2_arr, n_bins=L)
result3_cnt, result3_intensity = intensity_hist(result3_arr, n_bins=L)
result4_cnt, result4_intensity = intensity_hist(result4_arr, n_bins=L)

utils.plot_hist("prob2c", 
                [sample2_intensity, sample2_cnt],
                [result3_intensity, result3_cnt],
                [result4_intensity, result4_cnt]
                )

# prob (d)
def poy_cumsum(arr):
    res = []
    cumsum = 0
    for a in arr:
        cumsum += a
        res.append(cumsum)
    res = np.array(res)
    return res

def global_hist_equal(img_arr):
    '''
    parameters:
        img_arr [np.array]: image with shape (m*n)
    note:
        1. symbol ref: [Histogram equalization](https://en.wikipedia.org/wiki/Histogram_equal)
    '''
    L = 256
    M, N = img_arr.shape
    cnt, intensity = intensity_hist(img_arr, L)
    pdf = cnt / (M * N)
    cdf = poy_cumsum(pdf)
    s = utils.int_round( (L - 1) * cdf )
    def T(r, s):
        return s[r]
    res = T(img_arr, s)
    return res

result5_arr = global_hist_equal(sample3_arr)
utils.save_npArr2JPG(result5_arr, "5_result")

# prob (e)
def hist_equal_for_kernel(img_arr):
    '''
    parameters:
        img_arr [np.array]: image with shape (m*n)
    note:
        1. symbol ref: [Histogram equalization](https://en.wikipedia.org/wiki/Histogram_equal)
    '''
    L = 256
    M, N = img_arr.shape
    cnt, intensity = intensity_hist(img_arr, L)
    pdf = cnt / (M * N)
    cdf = poy_cumsum(pdf)
    s = np.array( np.round( (L - 1) * cdf), dtype=np.uint8 )
    def T(r, s):
        return s[r]
    res = T(img_arr[M//2, N//2], s)
    return res

def local_hist_equal(img_arr, kernel_size=51, pad_method="edge"):
    M, N = img_arr.shape
    result = np.zeros( (M, N) )
    n_pad = kernel_size // 2
    img_expand_arr = np.pad(img_arr, ((n_pad, n_pad), (n_pad, n_pad)), "edge")
    # make list of input first
    result_list = []
    for i in range(M):
        for j in range(N):
            result_list.append(img_expand_arr[i: i + kernel_size, j: j + kernel_size])
    # accelerate by multi-process
    with Pool(6) as p:
        result = p.map(hist_equal_for_kernel, result_list)
    
    result = np.array(result)
    result = result.reshape(M, N)
    return result

# Remember delete it before submit hw1
result6_arr = local_hist_equal(sample3_arr)
utils.save_npArr2JPG(result6_arr, "6_result")

# prob (f)
sample3_cnt, sample3_intensity = intensity_hist(sample3_arr, n_bins=L)
result5_cnt, result5_intensity = intensity_hist(result5_arr, n_bins=L)
result6_cnt, result6_intensity = intensity_hist(result6_arr, n_bins=L)
utils.plot_hist("prob2f",
                [sample3_intensity, sample3_cnt],
                [result5_intensity, result5_cnt],
                [result6_intensity, result6_cnt]
                )
# prob (g)
'''
- Linear scaling and clipping is **useless** as range is filling 0~255.
- Power-law
    - p=2: too dark
    - p=1/2: looks good!
- Rubber Band Error Code
    - cp=(0.5, 0.2): cool with some **dark aureole** but a little dark.
    - cp=(0.5, 0.8): cool but a little strange here! with *dark light*.
- Rubber Band
    - cp=(0.5, 0.2): too dark with clear light!
    - cp=(0.5, 0.8): looks good!
    - cp=(0.2, 0.5): looks good!
    - cp=(0.8, 0.2): too too dark...
    - cp=(0.2, 0.8): light is gone...
    - cp=(0.8, 0.2): too too dark...
- Logarithmic Point
    - a=1: looks good!
    - a=2: brighter than a=1.
    - a=1/2: dark but **unreasonable** for our range is not full of 0~1.
'''

# def transfer function

def transfer_linear(F_jk):
    max_F_jk, min_F_jk = np.max(F_jk), np.min(F_jk)
    G_jk = (1 - 0) / (max_F_jk - min_F_jk) * (F_jk - min_F_jk)
    G_jk[G_jk < 0] = 0
    G_jk[G_jk > 1] = 1
    return G_jk

def transfer_powerLaw(F_jk, p=2):
    G_jk = F_jk ** p
    return G_jk

def transfer_rubberBand_error_code(F_jk, cp=(0.5, 0.5)):
    max_F_jk, min_F_jk = np.max(F_jk), np.min(F_jk)
    slope1 = (cp[1] - 0) / (cp[0] - min_F_jk)
    slope2 = (1 - cp[1]) / (max_F_jk - cp[0])
    G_jk1 = slope1 * (F_jk - min_F_jk)
    G_jk2 = slope2 * (F_jk - cp[0])
    G_jk = np.where(F_jk <= cp[0], G_jk1, G_jk2)
    return G_jk

def transfer_rubberBand(F_jk, cp=(0.5, 0.5)):
    max_F_jk, min_F_jk = np.max(F_jk), np.min(F_jk)
    slope1 = (cp[1] - 0) / (cp[0] - min_F_jk)
    slope2 = (1 - cp[1]) / (max_F_jk - cp[0])
    G_jk1 = slope1 * (F_jk - min_F_jk)
    G_jk2 = slope2 * (F_jk - cp[0]) + cp[1]
    G_jk = np.where(F_jk <= cp[0], G_jk1, G_jk2)
    return G_jk

def transfer_generalRubberBand(F_jk, f1, f2, cp=(0.5, 0.5)):
    max_F_jk, min_F_jk = np.max(F_jk), np.min(F_jk)
    G_jk1 = f1(F_jk - min_F_jk)
    G_jk2 = f2(F_jk - cp[0]) + cp[1]
    G_jk = np.where(F_jk <= cp[0], G_jk1, G_jk2)
    return G_jk

def transfer_logPt(F_jk, a=1):
    G_jk = np.log(1 + a * F_jk) / np.log(2.0)
    return G_jk

def transfer_reverse(F_jk):
    G_jk = 1 - F_jk
    return G_jk

def transfer_inverse(F_jk):
    G_jk = np.where(F_jk > 0.1, 0.1 / F_jk, 1)
    return G_jk

def transfer_ampLVSlice(F_jk, Slice=(0.4, 0.6), level=0.6, Type="zero"):
    if Type == "zero":
        G_jk = np.where(np.logical_and(F_jk >= Slice[0], F_jk <= Slice[1]), level, 0)
    elif Type == "linear":
        G_jk = np.where(np.logical_and(F_jk >= Slice[0], F_jk <= Slice[1]), level, F_jk)
    return G_jk

def transfer_Okapi_BM25(F_jk, k=1):
    G_jk = (k + 1) * F_jk / (F_jk + k)
    return G_jk

sample4_F_jk = sample4_arr / 255

result7_1_arr = utils.int_round(transfer_powerLaw(sample4_F_jk, p=1/2)*255)
result7_2_arr = utils.int_round(transfer_rubberBand(sample4_F_jk, cp=(0.2, 0.5))*255)
result7_3_arr = utils.int_round(transfer_logPt(sample4_F_jk, a=1)*255)
result7_4_arr = utils.int_round(transfer_reverse(sample4_F_jk)*255)
result7_5_arr = utils.int_round(transfer_inverse(sample4_F_jk)*255)
result7_6_arr = utils.int_round(global_hist_equal(sample4_arr))
result7_7_arr = utils.int_round(transfer_ampLVSlice(sample4_F_jk, Slice=(0.2, 0.4), level=0.6, Type="zero")*255)
result7_8_arr = utils.int_round(transfer_Okapi_BM25(sample4_F_jk, k=0.01)*255)
result7_9_arr = utils.int_round(transfer_generalRubberBand(sample4_F_jk, 
                                           lambda X: transfer_powerLaw(X, 1/2),
                                           lambda X: transfer_powerLaw(X, 2),
                                           cp=(0.7, 0.83)
                                           )*255)

utils.plot_transfer("powerLaw", lambda X: transfer_powerLaw(X, p=1/2), "tmp")
utils.plot_transfer("rubberBand", lambda X: transfer_rubberBand(X, cp=(0.2, 0.5) ), "tmp")
utils.plot_transfer("logPt", lambda X: transfer_logPt(X, a=1), "tmp")
utils.plot_transfer("reverse", lambda X: transfer_reverse(X), "tmp")
utils.plot_transfer("inverse", lambda X: transfer_inverse(X), "tmp")
utils.plot_transfer("ampLVSlice", lambda X: transfer_ampLVSlice(X, Slice=(0.2, 0.4), level=0.6, Type="zero"), "tmp")
utils.plot_transfer("Okapi_BM25", lambda X: transfer_Okapi_BM25(X, k=0.01), "tmp")
GRB = lambda Y: transfer_generalRubberBand(Y, lambda X: transfer_powerLaw(X, 1/2), lambda X: transfer_powerLaw(X, 2), cp=(0.7, 0.83))
utils.plot_transfer("generalRubberBand", lambda X: GRB(X) )

utils.save_npArr2JPG(result7_1_arr, "tmp/7_result_1")
utils.save_npArr2JPG(result7_2_arr, "tmp/7_result_2")
utils.save_npArr2JPG(result7_3_arr, "tmp/7_result_3")
utils.save_npArr2JPG(result7_4_arr, "tmp/7_result_4")
utils.save_npArr2JPG(result7_5_arr, "tmp/7_result_5")
utils.save_npArr2JPG(result7_6_arr, "tmp/7_result_6")
utils.save_npArr2JPG(result7_7_arr, "tmp/7_result_7")
utils.save_npArr2JPG(result7_8_arr, "tmp/7_result_8")
utils.save_npArr2JPG(result7_9_arr, "7_result")

sample4_cnt, sample4_intensity = intensity_hist(sample4_arr, n_bins=L)
result7_cnt, result7_intensity = intensity_hist(result7_9_arr, n_bins=L)
utils.plot_hist("prob2g",
                [sample4_intensity, sample4_cnt],
                [result7_intensity, result7_cnt]
                )

