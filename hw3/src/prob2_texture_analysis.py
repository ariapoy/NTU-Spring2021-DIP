from PIL import Image
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

import utils

import pdb

import warnings
warnings.filterwarnings("ignore")
np.random.seed=0

# load image
sample2_arr = utils.load_img2npArr("sample2.png")

# prob (a)
"""
Perform Law’s method on sample2.png to obtain the feature vector of each pixel and discuss the feature vectors in your report.
"""
def laws_method(img_arr, window_size=13):
    F = img_arr
    if len(F.shape) > 2:
        F = F[:, :, 0]
    # Step 1 Convolution
    H1 = 1/36*np.array([[ 1, 2, 1],
                        [ 2, 4, 2],
                        [ 1, 2, 1]])
    H2 = 1/12*np.array([[ 1, 0,-1],
                        [ 2, 0,-2],
                        [ 1, 0,-1]])
    H3 = 1/12*np.array([[-1, 2,-1],
                        [-2, 4,-2],
                        [-1, 2,-1]])
    H4 = 1/12*np.array([[-1,-2,-1],
                        [ 0, 0, 0],
                        [ 1, 2, 1]])
    H5 =  1/4*np.array([[ 1, 0,-1],
                        [ 0, 0, 0],
                        [-1, 0, 1]])
    H6 =  1/4*np.array([[-1, 2,-1],
                        [ 0, 0, 0],
                        [ 1,-2, 1]])
    H7 = 1/12*np.array([[-1,-2,-1],
                        [ 2, 4, 2],
                        [-1,-2,-1]])
    H8 =  1/4*np.array([[-1, 0, 1],
                        [ 2, 0,-2],
                        [-1, 0, 1]])
    H9 =  1/4*np.array([[ 1,-2, 1],
                        [-2, 4,-2],
                        [ 1,-2, 1]])
    ## calculate M microstructure array
    M1 = convolve2d(F, H1, boundary='symm', mode='same')
    M2 = convolve2d(F, H2, boundary='symm', mode='same')
    M3 = convolve2d(F, H3, boundary='symm', mode='same')
    M4 = convolve2d(F, H4, boundary='symm', mode='same')
    M5 = convolve2d(F, H5, boundary='symm', mode='same')
    M6 = convolve2d(F, H6, boundary='symm', mode='same')
    M7 = convolve2d(F, H7, boundary='symm', mode='same')
    M8 = convolve2d(F, H8, boundary='symm', mode='same')
    M9 = convolve2d(F, H9, boundary='symm', mode='same')
    # Step 2 Energy computation
    S = np.ones( (window_size, window_size) )
    T1 = convolve2d(M1**2, S, boundary='symm', mode='same')
    T2 = convolve2d(M2**2, S, boundary='symm', mode='same')
    T3 = convolve2d(M3**2, S, boundary='symm', mode='same')
    T4 = convolve2d(M4**2, S, boundary='symm', mode='same')
    T5 = convolve2d(M5**2, S, boundary='symm', mode='same')
    T6 = convolve2d(M6**2, S, boundary='symm', mode='same')
    T7 = convolve2d(M7**2, S, boundary='symm', mode='same')
    T8 = convolve2d(M8**2, S, boundary='symm', mode='same')
    T9 = convolve2d(M9**2, S, boundary='symm', mode='same')
    result_arr = np.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9])
    return np.moveaxis(result_arr, 0, -1) 

prob2a = laws_method(sample2_arr, window_size=13)
prob2a_df = pd.DataFrame(prob2a.reshape(-1, 9))
prob2a_stats = prob2a_df.describe()

# prob (b)
"""
Use k-means algorithm to classify each pixel with the feature vectors you obtained from (a). Label same kind of texture with the same color and output it as result3.png
"""
def kMeans(X, K, maxIters=10):
    """
    Ref: [KMeans Clustering Implemented in python with numpy](https://gist.github.com/tvwerkhoven/4fdc9baad760240741a09292901d3abd)
    """
    centroids = X[np.random.choice(np.arange(len(X)), K)]
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Ensure we have K clusters, otherwise reset centroids and start over
        # If there are fewer than K clusters, outcome will be nan.
        if (len(np.unique(C)) < K):
            centroids = X[np.random.choice(np.arange(len(X)), K)]
        else:
            # Move centroids step 
            centroids = [X[C == k].mean(axis = 0) for k in range(K)]
    return np.array(centroids), C

# Remember uncomment it before submit
centroids3, label3_arr = kMeans(prob2a.reshape(-1, 9), K=5, maxIters=20)
result3_arr = label3_arr.reshape(prob2a.shape[:2])
fig, ax = plt.subplots()
cmap = plt.cm.RdBu
ax.matshow(result3_arr, interpolation='none', cmap=cmap)
fig.savefig("result3.png")
plt.clf()

# prob (c)
"""
Based on result3.png, design a method to improve the classification result and output the updated result as result4.png. Describe the modifications in detail and explain the reason why.
"""
prob2c = prob2a.reshape(-1, 9)
## position as features
row_fea = np.arange(prob2a.shape[0]).repeat(prob2a.shape[1]).reshape(prob2a.shape[:2]).reshape(-1, 1)
row_scalar = prob2a_stats.loc["mean"].mean()/200
row_fea = row_scalar*row_fea
#col_fea = np.arange(prob2a.shape[1]).repeat(prob2a.shape[0]).reshape(prob2a.shape[1::-1]).T.reshape(-1, 1)
prob2c = np.hstack([prob2c, row_fea])
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#prob2c_normal = scaler.fit_transform(prob2c)
#prob2c_normal = prob2c_normal[:, :-1] # only use row position
centroids4, label4_arr = kMeans(prob2c, K=5)
result4_arr = label4_arr.reshape(prob2a.shape[:2])

fig, ax = plt.subplots()
ax.matshow(result4_arr, interpolation='none', cmap=cmap)
fig.savefig("result4.png")
plt.clf()

