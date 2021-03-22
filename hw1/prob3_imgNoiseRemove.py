from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

import pdb

# useful utils
def save_npArr2JPG(img_arr, name):
    img = Image.fromarray(img_arr)
    img.convert("L").save("{0}.jpg".format(name), "JPEG")
    print("Save fig {0}.jpg!".format(name))

# prob (a)
'''
Solutions 
1. low-pass filtering (for uniform noise)
2. non-linear filtering (for impluse noise)
3. fft filtering [Image denoising by FFT](http://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html)

Question 1
How do I generate general form of low-pass filtering with higher kernel/mask size?
Ref: [For b=2, kernel size in (3, 5, 7)](https://www.researchgate.net/profile/Oleg-Shipitko/publication/325768087/figure/fig2/AS:637519863508992@1529007988866/Discrete-approximation-of-the-Gaussian-kernels-3x3-5x5-7x7.png)
Answer 1
Amy: You shouldn't follow the strict principle of example in lecture. You could follow this rule and design the matrix with flexible:
    1. Weighted average, so you need to sum as 1.
'''

def filter_low_pass(img_arr, b=1, kernel_size=3, pad_method="edge"):
    M, N = img_arr.shape
    result = np.zeros( (M, N) )
    n_pad = kernel_size // 2
    img_expand_arr = np.pad(img_arr, ((n_pad, n_pad), (n_pad, n_pad)), "edge")
    M_expand, N_expand = img_expand_arr.shape
    kernel_range = kernel_size
    
    # I fix kernel_size here as I couldn't create the general form of larger kernel size.
    # (b + k - 1)**2 with cross 
    #mask = 1 / ( (b + 2)**2 ) * np.array([[1, b, 1], [b, b**2, b], [1, b, 1]])
    mask = np.ones( (kernel_size, kernel_size)  )

    for i in range(-(kernel_size//2), (kernel_size//2)+1 ):
        mask[kernel_size // 2 + i, kernel_size // 2] = b
    for j in range(-(kernel_size//2), (kernel_size//2)+1 ):
        mask[kernel_size // 2, kernel_size // 2 + j] = b
    mask[kernel_size // 2, kernel_size // 2] = b ** 2
    mask = 1 / ( (b + kernel_size - 1)**2 ) * mask
    #print(mask)

    for i in range(M):
        for j in range(N):
            result_part = img_expand_arr[i: i + kernel_range, j: j + kernel_range] * mask
            # some problems here!
            result[i, j] = np.sum(result_part)
    return result

def filter_median(img_arr, kernel_size=3, pad_method="edge"):
    M, N = img_arr.shape
    result = np.zeros( (M, N) )
    n_pad = kernel_size // 2
    img_expand_arr = np.pad(img_arr, ((n_pad, n_pad), (n_pad, n_pad)), "edge")
    M_expand, N_expand = img_expand_arr.shape
    kernel_range = kernel_size
    
    for i in range(M):
        for j in range(N):
            result_part = img_expand_arr[i: i + kernel_range, j: j + kernel_range]
            result[i, j] = np.median(result_part)
    return result

def filter_fft(img_arr, frac=0.1):
    result = fftpack.fft2(img_arr)
    keep_fraction = frac
    r, c = result.shape
    result[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    result[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    return result

def plot_spectrum(im_fft, fig_name):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.figure()
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.title('Fourier transform')
    plt.savefig("Fourier transform of {0}".format(fig_name))
    plt.clf()

sample6 = Image.open("sample6.jpg")
sample6_arr = np.array(sample6)

sample7 = Image.open("sample7.jpg")
sample7_arr = np.array(sample7)

result8_3_fft_arr = filter_fft(sample6_arr)
plot_spectrum(result8_3_fft_arr, "sample6")
result8_3_arr = fftpack.ifft2(result8_3_fft_arr).real
save_npArr2JPG(result8_3_arr, "8_result_fft")

result9_3_fft_arr = filter_fft(sample7_arr)
result9_3_arr = fftpack.ifft2(result9_3_fft_arr).real
save_npArr2JPG(result9_3_arr, "9_result_fft")

result8_1_arr = filter_low_pass(sample6_arr, b=2, kernel_size=3)
save_npArr2JPG(result8_1_arr, "8_result_low_pass")

result8_2_arr = filter_median(sample6_arr, kernel_size=5)
save_npArr2JPG(result8_2_arr, "8_result_median")

result9_1_arr = filter_low_pass(sample7_arr, b=2, kernel_size=3)
save_npArr2JPG(result9_1_arr, "9_result_low_pass")

result9_2_arr = filter_median(sample7_arr, kernel_size=5)
save_npArr2JPG(result9_2_arr, "9_result_median")

# prob (b)
'''
Question 2
Can I use `np.prod`?
'''

def PSNR(F_true, F):
    MSE = np.sum((F_true - F)**2) / np.prod(F.shape)
    res = 10 * np.log10(255**2 / MSE)
    return res

sample5 = Image.open("sample5.jpg")
sample5_arr = np.array(sample5)

print('PSNR of random noise by fft : {0}'.format(PSNR(sample5_arr, result8_1_arr)))
print('PSNR of random noise by low-pass : {0}'.format(PSNR(sample5_arr, result8_1_arr)))
print('PSNR of random noise by median   : {0}'.format(PSNR(sample5_arr, result8_2_arr)))
print('PSNR of impulse noise by fft: {0}'.format(PSNR(sample5_arr, result9_1_arr)))
print('PSNR of impulse noise by low-pass: {0}'.format(PSNR(sample5_arr, result9_1_arr)))
print('PSNR of impulse noise by median  : {0}'.format(PSNR(sample5_arr, result9_2_arr)))

