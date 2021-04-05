from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

import utils

import pdb

# load image
sample1_arr = utils.load_img2npArr("sample1.jpg")
sample2_arr = utils.load_img2npArr("sample2.jpg")

# prob (a)

## prob (a-1)
'''
Perform 1st order edge detection and output the edge maps as result1.jpg
'''

def first_order_detection(img_arr, T=None, n_points=9, k=2, check_row_col=None, verbose="prob1_a_1_grad_hist"):
    """1st order edge detection
    img_arr: np.array with img_arr.shape=(img.height, img.width)
        Grayscale image as NumPy array with range [0, 255]
    T: float
        Threshold of judging point (j, k) as edge point
        if G(j, k) >= T then set it as edge point.
    n_points: int in {3, 4, 9}
        type of mask array, please see Lec 3 page 12 -- 15
    k: int
        scale when n_points=9
        - when k=1, it's called "Prewitt Mask"
        - when k=2, it's called "Sobel Mask"
    """
    # Step 1. Compute edge magnitude
    # expand array
    M, N = img_arr.shape
    kernel_size = 3 # For gradient.
    n_pad = kernel_size // 2
    img_expand_arr = np.pad(img_arr, ((n_pad, n_pad), (n_pad, n_pad)), "edge")
    G_row = np.zeros( (M, N) )
    G_col = np.zeros( (M, N) )
    # build mask/filter
    if n_points == 3:
        row_mask = np.array([[0, 0, 0], [-1, 1, 0], [0,  0, 0]])
        col_mask = np.array([[0, 0, 0], [ 0, 1, 0], [0, -1, 0]])
    elif n_points == 4:
        row_mask = np.array([[0, 0, 0], [-1, 1, 0], [0,  0, 0]])
        col_mask = np.array([[0, 0, 0], [ 0, 1, 0], [0, -1, 0]])
    elif n_points == 9:
        row_mask = 1 / (k + 2) * np.array([[-1, 0, 1], [-k, 0, k], [-1,  0,  1]])
        col_mask = 1 / (k + 1) * np.array([[ 1, k, 1], [ 0, 0, 0], [-1, -k, -1]])
    else:
        print("Error type! Please enter one of {3, 4, 9}.")
        return None
    # convolution/weighted average
    """
    for i in range(M):
        for j in range(N):
            G_row[i, j] = np.sum(img_expand_arr[i: i + kernel_size, j: j + kernel_size] * row_mask)
            G_col[i, j] = np.sum(img_expand_arr[i: i + kernel_size, j: j + kernel_size] * col_mask)
    """
    G_row = convolve2d(img_arr, row_mask, boundary='symm', mode='same')
    G_col = convolve2d(img_arr, col_mask, boundary='symm', mode='same')
    G_grad = np.sqrt(G_row ** 2 + G_col ** 2)
    if n_points == 4:
        theta = np.arctan(G_col / G_row) + np.pi / 4
    else:
        theta = np.arctan(G_col / G_row)
    # Analyze the statistics of the magnitude.
    # Examine the cumulative distribution function
    #G_grad_max, G_grad_min = np.max(G_grad), np.min(G_grad)
    #print("Range of magnitudes: ({1}, {0})".format(G_grad_max, G_grad_min) )
    #bins = np.arange(G_grad_min, G_grad_max)
    #G_cnt, gradients = utils.poy_histogram(G_grad, bins)
    #utils.plot_hist("tmp/prob1_a_1_grad_hist", [gradients, G_cnt])
    cnt, intensity = np.histogram(G_grad)
    plt.plot(intensity[:-1], cnt) #, density=True, bins=bins)
    plt.savefig( "{0}.png".format("tmp/".format(verbose) ) )
    plt.clf()
    # Check row and col
    if check_row_col == "row":
        G_grad = G_row
    if check_row_col == "col":
        G_grad = G_col
    # Step 2. Threshold
    if T is not None:
        edge_map_arr = np.where(G_grad >= T, 255, 0)
    else:
        edge_map_arr = G_grad
    return edge_map_arr, theta

# Remember uncomment it before you submit.
"""
result1_arr, _ = first_order_detection(sample1_arr, T=2, n_points=9, k=2)
utils.save_npArr2JPG(result1_arr, "result1_T=2")
result1_arr, _ = first_order_detection(sample1_arr, T=25, n_points=9, k=2, check_row_col="row")
utils.save_npArr2JPG(result1_arr, "result1_row")
result1_arr, _ = first_order_detection(sample1_arr, T=25, n_points=9, k=2, check_row_col="col")
utils.save_npArr2JPG(result1_arr, "result1_col")
result1_arr, _ = first_order_detection(sample1_arr, T=25, n_points=9, k=2)
utils.save_npArr2JPG(result1_arr, "result1")
"""

# prob (a-2)
"""
Perform 2nd order edge detection and output the edge maps as result2.jpg
"""
def second_order_detection(img_arr, T, n_neighbor=8, type_nb="non-separable"):
    """2nd order edge detection
    img_arr: np.array with img_arr.shape=(img.height, img.width)
        Grayscale image as NumPy array with range [0, 255]
    T: float
        Threshold to separate zero and non-zero of Laplacian.
        if G(j, k) <= T then set G'(j, k) = 0.
    n_neighbor: int in {4, 8}
        number of neighbor for mask array, please see Lec 3 page 29
    type_nb: str in {"non-separable", "separable", "H1", "H2"}
        types of 8-neighbor mask array
    """
    # Step 1. Compute Laplacian
    # expand array
    M, N = img_arr.shape
    kernel_size = 3 # For gradient.
    n_pad = kernel_size // 2
    img_expand_arr = np.pad(img_arr, ((n_pad, n_pad), (n_pad, n_pad)), "edge")
    G_Laplacian = np.zeros( (M, N) )
    # build mask/filter
    if n_neighbor == 4:
        mask = 1/4 * np.array([ [0, -1, 0], [-1, 4, -1], [0,  -1, 0] ] )
    elif n_neighbor == 8:
        if type_nb == "non-separable":
            mask = 1/8 * np.array( [ [-1, -1, -1], [-1, 8, -1], [-1, -1, -1] ] )
        elif type_nb == "separable":
            mask = None
        elif type_nb == "H1":
            mask = None
        elif type_nb == "H2":
            mask = None
        else:
            print("Error value! Please enter one of {'non-separable', 'separable', 'H1', 'H2'}")
    else:
        print("Error value! Please enter one of {4, 8}.")
        return None
    # convolution/weighted average
    """
    for i in range(M):
        for j in range(N):
            G_Laplacian[i, j] = np.sum(img_expand_arr[i: i + kernel_size, j: j + kernel_size] * mask)
    """
    G_Laplacian = convolve2d(img_arr, mask, boundary='symm', mode='same')
    # Step 2. Zero-crossing detection
    # Generation the histogram of L
    #G_Laplacian_max, G_Laplacian_min = np.max(G_Laplacian), np.min(G_Laplacian)
    #print("Range of Laplacian: ({1}, {0})".format(G_Laplacian_max, G_Laplacian_min) )
    #bins = np.arange(G_Laplacian_min, G_Laplacian_max)
    #G_cnt, gradients = utils.poy_histogram(G_Laplacian, bins)
    #utils.plot_hist("tmp/prob1_a_2_laplacian_hist", [gradients, G_cnt])
    cnt, intensity = np.histogram(G_Laplacian)
    plt.plot(intensity[:-1], cnt) #, density=True, bins=bins)
    plt.savefig( "{0}.png".format("tmp/prob1_a_2_laplacian_hist") )
    plt.clf()
    # Set up a threshold to separate zero and non-zero, output as GG
    GG = np.where(np.abs(G_Laplacian) <= T, 0, G_Laplacian)
    GG = np.sign(GG)
    # For GG = 0, decide whether (j, k) is a zero-crossing point
    GG_expand_arr = np.pad(GG, ((n_pad, n_pad), (n_pad, n_pad)), "edge")
    # mask for {-1, 1} in the 8-neighbor of zero-crossing point
    # make center point as two times of -1 0 1 => -2 0 2
    #mask_zeroCrossing = np.array( [[1, 1, 1], [1, 0, 1], [1, 1, 1]] )
    is_edge_arr = np.zeros( (M, N) )
    for i in range(M):
        for j in range(N):
            center = GG_expand_arr[ (i + kernel_size)//2, (j + kernel_size)//2 ]
            if center == 0:
                # possible cross point is {-1, 0, 1}
                crosspt = np.unique(GG_expand_arr[i: i + kernel_size, j: j + kernel_size] )
                if (-1 in crosspt) & (1 in crosspt):
                    is_edge_arr[i, j] = 1
                else:
                    is_edge_arr[i, j] = 0
            else:
                is_edge_arr[i, j] = 0
    # As is_edge_arr in the range of {-1, 0, 1, +-2}
    # The edge occurs at {-1, 0, 1}
    edge_map_arr = np.where(is_edge_arr == 1, 255, 0)
    return edge_map_arr

# Remember uncomment it before you submit.
"""
#result2_arr = second_order_detection(sample1_arr, 50, n_neighbor=8, type_nb="non-separable")
#utils.save_npArr2JPG(result2_arr, "result2_T50")
#result2_arr = second_order_detection(sample1_arr, 25, n_neighbor=8, type_nb="non-separable")
#utils.save_npArr2JPG(result2_arr, "result2_Y25")
result2_arr = second_order_detection(sample1_arr, 5, n_neighbor=8, type_nb="non-separable")
utils.save_npArr2JPG(result2_arr, "result2")
"""

# prob (a-3)
"""
Perform Canny edge detection and output the edge maps as result3.jpg
"""
def Canny_edge_detection(img_arr, TH, TL, n_neighbor=8):
    """
    Canny Edge Detector
    """
    M, N = img_arr.shape
    # Step 1 Noise reduction
    Gaussian_filter = 1/159 * np.array([[2, 4, 5, 4, 2], 
                                        [4, 9,12, 9, 4],
                                        [5,12,15,12, 5],
                                        [4, 9,12, 9, 4],
                                        [2, 4, 5, 4, 2]]
                                      )
    Fjk = convolve2d(img_arr, Gaussian_filter, boundary='symm', mode='same')
    # Step 2 Compute gradient magnitude and orientation
    Gjk, theta = first_order_detection(Fjk, T=None, n_points=9, k=2, check_row_col=None)
    # Step 3 Non-maximal suppression
    GNjk = np.zeros( (M, N) )
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # theta in (-22.5, 22.5) or theta in (157.5, -157.5)
            if ( (theta[i, j] >= -np.pi/8) & (theta[i, j] <= np.pi/8) ) or ( (theta[i, j] >= 7*np.pi/8) & (theta[i, j] <= -7*np.pi/8) ):
                if (Gjk[i, j] > Gjk[i, j-1]) & (Gjk[i, j] > Gjk[i, j+1] ):
                    GNjk[i, j] = Gjk[i, j]
            # theta in (22.5, 67.5) or theta in (-157.5, -112.5)
            elif ( (theta[i, j] > np.pi/8) & (theta[i, j] <= 3*np.pi/8) ) or ( (theta[i, j] >= -7*np.pi) & (theta[i, j] <= -5/np.pi*8) ):
                if (Gjk[i, j] > Gjk[i-1, j+1]) & (Gjk[i, j] > Gjk[i+1, j-1] ):
                    GNjk[i, j] = Gjk[i, j]
            # theta in (67.5, 112.5) or theta in (-112.5, -67.5)
            elif ( (theta[i, j] > 3*np.pi/8) & (theta[i, j] <= 5*np.pi/8) ) or ( (theta[i, j] >= -5*np.pi) & (theta[i, j] <= -3/np.pi*8) ):
                if (Gjk[i, j] > Gjk[i-1, j]) & (Gjk[i, j] > Gjk[i+1, j] ):
                    GNjk[i, j] = Gjk[i, j]
            # theat in (112.5, 157.5) or theta in (-)
            else:
                if (Gjk[i, j] > Gjk[i-1, j-1]) & (Gjk[i, j] > Gjk[i+1, j+1] ):
                    GNjk[i, j] = Gjk[i, j]
    # Hysteretic thresholding
    # 2 is edge pixel, 1 is candidate pixel, 0 is non-edge pixel
    is_edge_arr = np.where(GNjk >= TH, 2, 0)
    is_edge_arr = np.where( (GNjk >= TL) & (GNjk < TH), 1, is_edge_arr)
    # Connected component labeling method
    edge_map_arr = np.zeros( (M, N) )
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if is_edge_arr[i, j] == 2:
                edge_map_arr[i, j] = 255
            elif is_edge_arr[i, j] == 1:
                neighbors = np.zeros( (3, 3) )
                if n_neighbor == 8:
                    for krow in range(-1, 1+1):
                        for kcol in range(-1, 1+1):
                            if krow != kcol:
                                neighbors[krow, kcol] = is_edge_arr[i + krow, j + kcol]
                elif n_eighbor == 4:
                    pass
                if (neighbors >= 1).any():
                    edge_map_arr[i, j] = 255
            else:
                pass
    return edge_map_arr, Fjk, Gjk, GNjk, is_edge_arr

# Remember uncomment it before you submit.
"""
result3_arr, result3_s1_arr, result3_s2_arr, result3_s3_arr, result3_s4_arr = Canny_edge_detection(sample1_arr, 50, 15)
utils.save_npArr2JPG(result3_arr, "result3")
result3_s1_arr = utils.int_round(result3_s1_arr)
result3_s2_arr = np.where(result3_s2_arr > 25, 255, 0)
result3_s3_arr = np.where(result3_s3_arr > 25, 255, 0)
result3_s4_arr = np.where(result3_s4_arr >= 2, 255, 0)
result3_s4_arr = np.where( (result3_s4_arr >= 1) & (result3_s4_arr < 2), 127, result3_s4_arr)
utils.save_npArr2JPG(result3_s1_arr, "result3_s1")
utils.save_npArr2JPG(result3_s2_arr, "result3_s2")
utils.save_npArr2JPG(result3_s3_arr, "result3_s3")
utils.save_npArr2JPG(result3_s4_arr, "result3_s4")
"""

# prob (a-4)
"""
Apply an edge crispening method to the given image, and output the result as result4.jpg. Please also generate an edge map of result4.jpg as result5.jpg.
"""

def edge_crispen(img_arr, method="unsharp", c=3/5, L=7):
    # unsharp masking
    # In Lec 3 page 6
    if method == "unsharp":
        # Step 1. Low-pass filtering
        # where c in (3/5, 5/6), L is size of low pass filter
        lowPass_filter = 1/(1 + L - 1 )**2 * np.ones( (L, L) )
        FLjk = convolve2d(img_arr, lowPass_filter, boundary='symm', mode='same')
        Gjk = c/(2*c-1) * img_arr - (1-c)/(2*c-1) * FLjk
    # high pass filter
    # In Lec 3 page 5
    elif method == "H1":
        mask = np.array([[ 0,-1, 0],
                         [-1, 5,-1],
                         [ 0,-1, 0 ]])
        Gjk = convolve2d(img_arr, mask, boundary='symm', mode='same')
    elif method == "H2":
        mask = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        Gjk = convolve2d(img_arr, mask, boundary='symm', mode='same')
    return Gjk

# Remember uncomment it before you submit.
"""
result4_arr = edge_crispen(sample1_arr, method="unsharp", c=3/5, L=7)
result5_arr, _, _, _, _ = Canny_edge_detection(result4_arr, 50, 15)
utils.save_npArr2JPG(result4_arr, "result4")
utils.save_npArr2JPG(result5_arr, "result5")
"""

"""
# prob (a-5)
Compare result1.jpg, result2.jpg, result3.jpg and result5.jpg. Provide some discussions and findings in the report.
"""

# prob (b)
"""
Please design an algorithm to obtain the edge map of sample2.jpg as best as you can. Describe the steps in detail and provide some discussions.
"""

# Steps

def prob1b_edge_detection(img_arr):
    """exp1
    # Step 1. high and low intensity detailed image.
    GjkHigh = utils.transfer_powerLaw(img_arr, p=2)
    GjkLow = utils.transfer_powerLaw(img_arr, p=1/2)
    # Step 2. Canny edge detection for high and low intensity detailed image.
    _, _, _, _, GjkHigh_cand = Canny_edge_detection(GjkHigh, 60, 30, n_neighbor=8)
    _, _, _, _, GjkLow_cand = Canny_edge_detection(GjkLow, 60, 30, n_neighbor=8)
    # Step 3. Combine them with linear combination
    Gjk_cand = GjkHigh_cand + GjkLow_cand
    edge_map_arr = np.where(Gjk_cand >= 2, 255, 0)
    """

    # Step 1. high and low intensity detailed image.
    GjkLow = utils.transfer_powerLaw(img_arr, p=1.4)
    # Step 2. Canny edge detection for high and low intensity detailed image.
    _, _, _, _, Gjk_candidate = Canny_edge_detection(GjkLow, 60, 30, n_neighbor=8)
    Gjk = np.where(Gjk_candidate >= 2, 255, 0)
    Gjk = np.where( (Gjk_candidate >= 1) & (Gjk_candidate < 2), 127, Gjk)
    # Step 3. Combine them with linear combination
    """exp2
    L = 3
    lowPass_filter = 1/(1 + L - 1 )**2 * np.ones( (L, L) )
    FLjk = convolve2d(Gjk, lowPass_filter, boundary='symm', mode='same')
    c = 5/6
    edge_map_arr = c/(2*c-1) * Gjk - (1-c)/(2*c-1) * FLjk
    """
    edge_map_arr = Gjk
    return edge_map_arr

prob1b_arr = prob1b_edge_detection(sample2_arr)
utils.save_npArr2JPG(prob1b_arr, "prob1b")

