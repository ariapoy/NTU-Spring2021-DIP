from PIL import Image
import numpy as np
from numpy.linalg import matrix_power
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

import utils

import pdb

import warnings
warnings.filterwarnings("ignore")

# reproducible
SEED = 0
np.random.seed(SEED)

sample1_arr = utils.load_img2npArr("sample1.png")
print(sample1_arr.shape)

# prob 1(a)
"""
Perform dithering using the dither matrix \(I_{2}\) in Figure 1.(b) and output the result as \textbf{result1.png}
"""

def expandonce_dither_mat(I):
    I_expand_list = [4*I+(s%4) for s in range(1, 4+1)]
    I_expand = np.block([I_expand_list[:2], I_expand_list[2:]])
    return I_expand

def init_dither_mat(I, d=2):
    n_step = int(np.log2(d) )
    I_expand = I
    if n_step == 1:
        pass
    else:
        #I_expand_list = [dim*I+(s%dim) for s in range(1, dim+1)]
        #n_row = int(dim**(1/2))
        #I_expand = np.block([I_expand_list[:n_row], I_expand_list[n_row:]])
        for i in range(n_step - 1):
            I_expand = expandonce_dither_mat(I_expand)
    return I_expand

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
    F = img_arr
    # step1 add noise
    N = 255*np.random.normal(0, 0.05, F.shape)
    H = F + N
    H = np.where(H > 255, 255, H)
    H = np.where(H < 0, 0, H)
    H = utils.int_round(H)
    # step2 determine the threshold matrix
    T = 255*(I + 0.5)/(I.shape[0])**2
    T = utils.int_round(T)
    # step3 dither with (repeated) threshold matrix
    if H.shape != T.shape:
        scale = int(H.shape[0]/T.shape[0] )
        T_expand = np.tile(T, (scale, scale))
        G = np.where(H >= T_expand, 255, 0)
    else:
        G = np.where(H >= T, 255, 0)
    return G

dither_mat = init_dither_mat(np.array([[1, 2], 
                                       [3, 0]])
                             )
result1_arr = dithering(sample1_arr, dither_mat, noise_type="white")
utils.save_npArr2JPG(result1_arr, "result1")
print("Finish prob 1 (a).")

# prob 1(b)
"""
Expand the dither matrix \(I_{2}\) to \(I_{256}\) \((256 \times 256)\) and use it to perform dithering. Output the result as \textbf{result2.png}. Compare \textbf{result1.png} and \textbf{result2.png} along with some discussions.
"""

dither256_mat = init_dither_mat(dither_mat, d=256)
result2_arr = dithering(sample1_arr, dither256_mat, noise_type="white")
utils.save_npArr2JPG(result2_arr, "result2")
print("Finish prob 1 (b).")

# prob 1(c)
"""
Perform error diffusion with two different filter masks. Output the results as \textbf{result3.png}, and \textbf{result4.png}, respectively. Discuss these two masks based on the results. \\
"""

def err_diffusion(img_arr, filter_mask="Floyd", thr=0.5, n_iter=1):
    """
    Step 1 normalize F bet [0, 1], threshold = 0.5, filter_mask
    Step 2 Calculate G by threshold
    Step 3 Calculate error E = \tilde{F} - G
    Step 4 Error diffusion + serpentine scanning
    """
    # Step 1
    F = img_arr / img_arr.max()
    tildeF = F.copy()
    T = thr
    G = np.where(F >= T, 1, 0)
    E = F - G
    if filter_mask == "Floyd":
        mask = 1/16*np.array([
                              [0, 0, 7], 
                              [3, 5, 1]
                             ])
        n_pad = 1
    elif filter_mask == "Jarvis":
        mask = 1/48*np.array([
                              [0, 0, 0, 7, 5],
                              [3, 5, 7, 5, 3],
                              [1, 3, 5, 3, 1]
                             ])
        n_pad = 2
    else:
        mask = 1/2*np.array([[0, 1],
                             [1, 0],
                            ])
        n_pad = 1
    M, N = F.shape
    for i in range(0, M - n_pad):
        if i % 2 == 1:
            F[i:i+n_pad+1, :] = F[i:i+n_pad+1, ::-1]
            tildeF[i:i+n_pad+1, :] = tildeF[i:i+n_pad+1, ::-1]
            mask == mask[:, ::-1]
        for j in range(1, N - n_pad):
            tildeF[i:i+n_pad+1, j-n_pad:j+n_pad+1] = F[i:i+n_pad+1, j-n_pad:j+n_pad+1] + E[i,j] * mask
        if i % 2 == 1:
            F[i:i+n_pad+1, :] = F[i:i+n_pad+1, ::-1]
            tildeF[i:i+n_pad+1, :] = tildeF[i:i+n_pad+1, ::-1]
            mask == mask[:, ::-1]

    G = np.where(tildeF >= T, 1, 0)
    return utils.int_round(255*G)

result3_arr = err_diffusion(sample1_arr, filter_mask="Floyd", thr=0.5, n_iter=1)
utils.save_npArr2JPG(result3_arr, "result3")
#result4_arr = err_diffusion(sample1_arr, filter_mask="Jarvis", thr=0.5, n_iter=1)
#utils.save_npArr2JPG(result4_arr, "result4")
print("Finish prob 1 (c).")

