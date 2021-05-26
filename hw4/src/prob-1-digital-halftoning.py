from PIL import Image, ImageDraw
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

def err_diffusion(img_arr, filter_mask="Floyd", thres=0.5):
    F = img_arr / 255
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
        mask = 1/2*np.array([
                             [0, 0, 1],
                             [0, 1, 0],
                            ])
        n_pad = 1

    # serpentine scanning
    for i in range(0, F.shape[0] - n_pad):
        if i % 2 == 0:
            for j in range(n_pad, F.shape[1] - n_pad):
                Fij = F[i, j]
                Gij = np.where(Fij >= thres, 1, 0)
                F[i, j] = Gij
                Eij = Fij - Gij
                F[i:i+n_pad+1, j-n_pad:j+n_pad+1] += Eij*mask
        else:
            for j in range(F.shape[1] - n_pad - 1, n_pad - 1, -1):
                Fij = F[i, j]
                Gij = np.where(Fij >= thres, 1, 0)
                F[i, j] = Gij
                Eij = Fij - Gij
                F[i:i+n_pad+1, j-n_pad:j+n_pad+1] += Eij*mask[:, ::-1]

    G = np.where(F > thres, 1, 0)

    return utils.int_round(255*G)

result3_arr = err_diffusion(sample1_arr, thres=0.5)
utils.save_npArr2JPG(result3_arr, "result3")
result4_arr = err_diffusion(sample1_arr, filter_mask="Jarvis", thres=0.5)
utils.save_npArr2JPG(result4_arr, "result4")

result1c_diag_arr = err_diffusion(sample1_arr, filter_mask="diagonal", thres=0.5)
utils.save_npArr2JPG(result1c_diag_arr, "tmp/result1c_diag")

print("Finish prob 1 (c).")

# prob 1(c)
"""
Try to transfer \textbf{result1.png} to a dotted halftone/manga style binary image such as \textbf{sample1\_dotted.png} in Figure 1.(c). Describe the steps in detail and show the result. \\
"""

def dotted_style(img, radius=3):
    M, N = img.shape
    res_img_arr = np.zeros_like(img.copy())
    res_img = Image.fromarray(res_img_arr.astype(np.uint8))
    draw = ImageDraw.Draw(res_img)
    F = img/255
    for i in range(0, M, 2*radius+1):
    #for i in range(0, M - 2*radius):
    #for i in range(0, M - radius, radius+1):
        for j in range(0, N, 2*radius+1):
        #for j in range(0, N - 2*radius):
        #for j in range(0, N - radius, radius+1):
            i_bnd, j_bnd = i + 2*radius - 1, j + 2*radius - 1
            i_center, j_center = i + radius - 1, j + radius - 1
            Fij = F[i:i_bnd, j:j_bnd]
            Nij = np.random.normal(0, 0.05, Fij.shape)
            Hij = (Fij + Nij).mean()
            if Hij <= 0.2:
                radius_scale = 0
            elif Hij <= 0.4:
                radius_scale = radius - 2
            elif Hij <= 0.6:
                radius_scale = radius - 1
            elif Hij <= 0.8:
                radius_scale = radius
            else:
                radius_scale = radius + 1
            draw.ellipse([(j_center - radius_scale, i_center - radius_scale), (j_center + radius_scale, i_center + radius_scale)], fill=255, width=0)

    return np.array(res_img)

sample1_dotted_arr = dotted_style(result1_arr)
utils.save_npArr2JPG(sample1_dotted_arr, "sample1_dotted")
print("Finish prob 1 (d).")

