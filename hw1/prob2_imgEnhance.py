from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

import pdb

# prob (a)
sample2 = Image.open("sample2.jpg")
sample2_arr = np.array(sample2)

def save_npArr2JPG(img_arr, name):
    img = Image.fromarray(img_arr)
    img.convert("L").save("{0}.jpg".format(name), "JPEG")
    print("Save fig {0}.jpg!".format(name))

result3_arr = np.array( np.round( sample2_arr / 5 ), dtype=np.uint8)
save_npArr2JPG(result3_arr, "3_result")
#result3 = Image.fromarray(result3_arr)
#result3.convert("L").save("3_result.jpg", "JPEG")

# prob (b)
result4_arr = np.array( np.round( result3_arr * 5 ), dtype=np.uint8)
save_npArr2JPG(result4_arr, "4_result")
#result4 = Image.fromarray(result4_arr)
#result4.convert("L").save("4_result.jpg", "JPEG")

# prob (c)
'''
Question 2
Can I use `np.histogram` and `matplotlib` ?
Amy: No, you can't use it!
'''

def poy_histogram(img_arr, bins):
    intensity = bins
    cnt = np.zeros(bins.shape)
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            cnt[ img_arr[i, j] ] += 1
    #for i in bins:
    #    cnt[i] = np.sum(img_arr == i)
    return cnt[:-1], intensity

def intensity_hist(img_arr, n_bins=256):
    bins = np.arange(n_bins + 1)
    #cnt, intensity = np.histogram(img_arr, bins=bins)
    cnt, intensity = poy_histogram(img_arr, bins=bins)
    return cnt, intensity[:-1]

L = 256
sample2_cnt, sample2_intensity = intensity_hist(sample2_arr, n_bins=L)
result3_cnt, result3_intensity = intensity_hist(result3_arr, n_bins=L)
result4_cnt, result4_intensity = intensity_hist(result4_arr, n_bins=L)

fig, (ax_sample2_hist, ax_result3_hist, ax_result4_hist) = plt.subplots(1, 3)
fig.set_size_inches(20, 8)
fig.suptitle("Compare histogram of sample2's family")
ax_bin = np.diff(np.arange(L + 1))
ax_sample2_hist.bar(sample2_intensity, sample2_cnt, width=ax_bin, align="edge")
ax_result3_hist.bar(result3_intensity, result3_cnt, width=ax_bin, align="edge")
ax_result4_hist.bar(result4_intensity, result4_cnt, width=ax_bin, align="edge")
# Use plt.hist is significantly slower than histogram>bar.
#ax_result3_hist.hist(result3_arr, bins=gray_bins)
#ax_result4_hist.hist(result4_arr, bins=gray_bins)
plt.savefig("prob2(c).png")
'''
My observation
1. sample2 & result4 are the same.
2. result3's intensity is smaller than others, and it has smaller dynamic range.
'''

# prob (d)
'''
Warning 1
I forget what is *local* and *global* histogram equalization.
Recall:
    1. [Histogram equalization](https://en.wikipedia.org/wiki/Histogram_equal)

Question 3
Can I use `np.cumsum`?
Amy: No, you can't use it!
'''

def poy_cumsum(arr):
    #res = np.array([np.sum(arr[:i + 1]) for i in range(arr.shape[0])])
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
    # cdf = np.cumsum(pdf)
    # It seems that we have numerical error between np and poy.
    cdf = poy_cumsum(pdf)
    s = np.array( np.round( (L - 1) * cdf), dtype=np.uint8 )
    def T(r, s):
        return s[r]
    res = T(img_arr, s)
    #res = T(img_arr[M//2, N//2], s)
    return res

sample3 = Image.open("sample3.jpg")
sample3_arr = np.array(sample3)
result5_arr = global_hist_equal(sample3_arr)
save_npArr2JPG(result5_arr, "5_result")
#result5 = Image.fromarray(result5_arr)
#result5.convert("L").save("5_result.jpg", "JPEG")

# clean former result
plt.clf()

#fig, (ax_sample3_hist, ax_result5_hist) = plt.subplots(1, 2)
#fig.set_size_inches(15, 8)
#fig.suptitle("Compare histogram of sample3's family")
#sample3_cnt, sample3_intensity = intensity_hist(sample3_arr, n_bins=L)
#result5_cnt, result5_intensity = intensity_hist(result5_arr, n_bins=L)
#ax_sample3_hist.bar(sample3_intensity, sample3_cnt, width=ax_bin, align="edge")
#ax_result5_hist.bar(result5_intensity, result5_cnt, width=ax_bin, align="edge")
#plt.savefig("prob2(f).png")

# prob (e)
'''
Question 4
- Can I use `np.pad` to expand my array?
- Which **pad**, **kernel_size** shoud we use?
'''

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
    # cdf = np.cumsum(pdf)
    # It seems that we have numerical error between np and poy.
    cdf = poy_cumsum(pdf)
    s = np.array( np.round( (L - 1) * cdf), dtype=np.uint8 )
    def T(r, s):
        return s[r]
    #res = T(img_arr, s)
    res = T(img_arr[M//2, N//2], s)
    return res

def multi_hist_equal(X, K=51):
    #return global_hist_equal(X)[K//2, K//2]
    return hist_equal_for_kernel(X)

def local_hist_equal(img_arr, kernel_size=51, pad_method="edge"):
    M, N = img_arr.shape
    result = np.zeros( (M, N) )
    n_pad = kernel_size // 2
    img_expand_arr = np.pad(img_arr, ((n_pad, n_pad), (n_pad, n_pad)), "edge")
    M_expand, N_expand = img_expand_arr.shape
    kernel_range = kernel_size
    # make list of input first
    result_list = []
    for i in range(M):
        for j in range(N):
            result_list.append(img_expand_arr[i: i + kernel_range, j: j + kernel_range])
    
    with Pool(6) as p:
        result = p.map(multi_hist_equal, result_list)
    #result = []
    #for k in tqdm(range(len(result_list))):
    #    result.append(multi_hist_equal(result_list[k]))
    result = np.array(result)
    result = result.reshape(M, N)
    return result

# Remember delete it before submit hw1
result6_arr = local_hist_equal(sample3_arr)
save_npArr2JPG(result6_arr, "6_result")

# prob (f)

# clean former result
plt.clf()

fig, (ax_sample3_hist, ax_result5_hist, ax_result6_hist) = plt.subplots(1, 3)
fig.set_size_inches(20, 8)
fig.suptitle("Compare histogram of sample3's family")
sample3_cnt, sample3_intensity = intensity_hist(sample3_arr, n_bins=L)
result5_cnt, result5_intensity = intensity_hist(result5_arr, n_bins=L)
result6_cnt, result6_intensity = intensity_hist(result6_arr, n_bins=L)
ax_sample3_hist.bar(sample3_intensity, sample3_cnt, width=ax_bin, align="edge")
ax_result5_hist.bar(result5_intensity, result5_cnt, width=ax_bin, align="edge")
ax_result6_hist.bar(result6_intensity, result6_cnt, width=ax_bin, align="edge")
plt.savefig("prob2(f).png")


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
- Reverse
'''

# def transfer function

def transfer_linear(img_arr):
    F_jk = img_arr / 255 # scaling
    max_F_jk, min_F_jk = np.max(F_jk), np.min(F_jk)
    G_jk = (1 - 0) / (max_F_jk - min_F_jk) * (F_jk - min_F_jk)
    G_jk[G_jk < 0] = 0
    G_jk[G_jk > 1] = 1
    res = np.array( np.round(G_jk * 255), dtype=np.uint8)
    return res

def transfer_powerLaw(img_arr, p=2):
    F_jk = img_arr / 255
    G_jk = F_jk ** p
    res = np.array( np.round(G_jk * 255), dtype=np.uint8)
    return res

def transfer_rubberBand_error_code(img_arr, cp=(0.5, 0.5)):
    F_jk = img_arr / 255
    max_F_jk, min_F_jk = np.max(F_jk), np.min(F_jk)
    slope1 = (cp[1] - 0) / (cp[0] - min_F_jk)
    slope2 = (1 - cp[1]) / (max_F_jk - cp[0])
    G_jk1 = slope1 * (F_jk - min_F_jk)
    G_jk2 = slope2 * (F_jk - cp[0])
    G_jk = np.where(F_jk <= cp[0], G_jk1, G_jk2)
    #G_jk[G_jk < 0] = 0
    #G_jk[G_jk > 1] = 1
    res = np.array( np.round(G_jk * 255), dtype=np.uint8)
    return res

def transfer_rubberBand(img_arr, cp=(0.5, 0.5)):
    F_jk = img_arr / 255
    max_F_jk, min_F_jk = np.max(F_jk), np.min(F_jk)
    slope1 = (cp[1] - 0) / (cp[0] - min_F_jk)
    slope2 = (1 - cp[1]) / (max_F_jk - cp[0])
    G_jk1 = slope1 * (F_jk - min_F_jk)
    G_jk2 = slope2 * (F_jk - cp[0]) + cp[1]
    G_jk = np.where(F_jk <= cp[0], G_jk1, G_jk2)
    #G_jk[G_jk < 0] = 0
    #G_jk[G_jk > 1] = 1
    res = np.array( np.round(G_jk * 255), dtype=np.uint8)
    return res

def transfer_logPt(img_arr, a=1):
    F_jk = img_arr / 255
    G_jk = np.log(1 + a * F_jk) / np.log(2.0)
    res = np.array( np.round(G_jk * 255), dtype=np.uint8)
    return res

def transfer_reverse(img_arr):
    F_jk = img_arr / 255
    G_jk = 1 - F_jk
    res = np.array( np.round(G_jk * 255), dtype=np.uint8)
    return res

def transfer_inverse(img_arr):
    F_jk = img_arr / 255
    #G_jk1 = 0.1 / F_jk
    G_jk = np.where(F_jk > 0.1, 0.1 / F_jk, 1)
    res = np.array( np.round(G_jk * 255), dtype=np.uint8)
    return res

def transfer_ampLVSlice(img_arr, Slice=(0.4, 0.6), level=0.6, Type="zero"):
    F_jk = img_arr / 255
    if Type == "zero":
        G_jk = np.where(np.logical_and(F_jk >= Slice[0], F_jk <= Slice[1]), level, 0)
    elif Type == "linear":
        G_jk = np.where(np.logical_and(F_jk >= Slice[0], F_jk <= Slice[1]), level, F_jk)
    res = np.array( np.round(G_jk * 255), dtype=np.uint8 )
    return res

def transfer_Okapi_BM25(img_arr, k=1):
    F_jk = img_arr / 255
    G_jk = (k + 1) * F_jk / (F_jk + k)
    res = np.array(np.round( G_jk * 255), dtype=np.uint8)
    return res

sample4 = Image.open("sample4.jpg")
sample4_arr = np.array(sample4)
result7_1_arr = transfer_powerLaw(sample4_arr, p=1/2)
result7_2_arr = transfer_rubberBand(sample4_arr, cp=(0.2, 0.5))
result7_3_arr = transfer_logPt(sample4_arr, a=1)
result7_4_arr = transfer_reverse(sample4_arr)
result7_5_arr = transfer_inverse(sample4_arr)
result7_6_arr = global_hist_equal(sample4_arr)
result7_7_arr = transfer_ampLVSlice(sample4_arr, Slice=(0.2, 0.4), level=0.6, Type="zero")
result7_8_arr = transfer_Okapi_BM25(sample4_arr, k=0.01)

save_npArr2JPG(result7_1_arr, "7_result_1")
save_npArr2JPG(result7_2_arr, "7_result_2")
save_npArr2JPG(result7_3_arr, "7_result_3")
save_npArr2JPG(result7_4_arr, "7_result_4")
save_npArr2JPG(result7_5_arr, "7_result_5")
save_npArr2JPG(result7_6_arr, "7_result_6")
save_npArr2JPG(result7_7_arr, "7_result_7")
save_npArr2JPG(result7_8_arr, "7_result_8")

# clean former result
plt.clf()
fig, (ax_sample4_hist, ax_result7_1_hist) = plt.subplots(1, 2)
fig.set_size_inches(16, 8)
fig.suptitle("Compare histogram of sample4's family")
sample4_cnt, sample4_intensity = intensity_hist(sample4_arr, n_bins=L)
result7_1_cnt, result7_1_intensity = intensity_hist(result7_1_arr, n_bins=L)
ax_sample4_hist.bar(sample4_intensity, sample4_cnt, width=ax_bin, align="edge")
ax_result7_1_hist.bar(result7_1_intensity, result7_1_cnt, width=ax_bin, align="edge")
plt.savefig("prob2(g).png")

