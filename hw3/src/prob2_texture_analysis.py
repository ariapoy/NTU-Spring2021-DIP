from PIL import Image
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from scipy.signal import convolve2d
from scipy import stats
import matplotlib.pyplot as plt
from collections import deque

import utils

import pdb

import warnings
warnings.filterwarnings("ignore")
# fix random
np.random.seed=0
# color of cluster
cmap = plt.cm.tab20

# load image
sample2_arr = utils.load_img2npArr("sample2.png")
sample3_arr = utils.load_img2npArr("sample3.png")

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
# Remember uncomment it before submit
prob2a_df = pd.DataFrame(prob2a.reshape(-1, 9))
print(prob2a_df.shape)
prob2a_stats = prob2a_df.describe()
print(prob2a_stats)
prob2a_stats.to_csv("tmp/prob2a_stats.csv")
ax_scatter = pd.plotting.scatter_matrix(prob2a_df, alpha=0.2, diagonal="kde")

plt.savefig("tmp/prob2a_scatter_matrix.png")
plt.clf()
ax = prob2a_stats.boxplot()
plt.savefig("tmp/prob2a_boxplot.png")
plt.clf()
for i in range(prob2a_df.shape[1]):
    ax = plt.subplot(3, 3, i+1)
    res = stats.probplot(prob2a_df.iloc[:, i], plot=plt)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
 
plt.savefig("tmp/prob2a_probplot.png")
plt.clf()

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
centroids3, label3_cluster_arr = kMeans(prob2a.reshape(-1, 9), K=4, maxIters=20)
result3_cluster_arr = label3_cluster_arr.reshape(prob2a.shape[:2])
#result3_arr = np.array( Image.new('RGB', result_3_cluster_arr.shape  ) )
result3_arr = np.array( Image.new('RGB', result3_cluster_arr.shape[::-1]  )  )
for i, cluster_id in enumerate(np.unique(result3_cluster_arr)):
    position = np.argwhere(result3_cluster_arr == cluster_id)
    result3_arr[position[:, 0], position[:, 1], :] = np.array(cmap(i)[:3])*255

utils.save_npArr2JPG(result3_arr, "result3")

centroids3_keq5, label3_keq5cluster_arr = kMeans(prob2a.reshape(-1, 9), K=5)
result3_keq5cluster_arr = label3_keq5cluster_arr.reshape(prob2a.shape[:2])
result3_keq5arr = np.array( Image.new('RGB', result3_keq5cluster_arr.shape[::-1]  )  )
for i, cluster_id in enumerate(np.unique(result3_keq5cluster_arr)):
    position = np.argwhere(result3_keq5cluster_arr == cluster_id)
    result3_keq5arr[position[:, 0], position[:, 1], :] = np.array(cmap(i)[:3])*255

utils.save_npArr2JPG(result3_keq5arr, "tmp/result3_keq5")

# prob (c)
"""
Based on result3.png, design a method to improve the classification result and output the updated result as result4.png. Describe the modifications in detail and explain the reason why.
"""
prob2c = prob2a.reshape(-1, 9)
## position as features
row_fea = np.arange(prob2a.shape[0]).repeat(prob2a.shape[1]).reshape(prob2a.shape[:2]).reshape(-1, 1)
row_scalar = prob2a_stats.loc["mean"].mean()/200
row_fea = row_scalar*row_fea
prob2c = np.hstack([prob2c, row_fea])
centroids4, result4_cluster_arr = kMeans(prob2c, K=4, maxIters=20)
result4_cluster_arr = result4_cluster_arr.reshape(prob2a.shape[:2])

result4_arr = np.array( Image.new('RGB', result4_cluster_arr.shape[::-1]  )  )
for i, cluster_id in enumerate(np.unique(result4_cluster_arr)):
    position = np.argwhere(result4_cluster_arr == cluster_id)
    result4_arr[position[:, 0], position[:, 1], :] = np.array(cmap(i)[:3])*255

utils.save_npArr2JPG(result4_arr, "result4")

# ehance of result 4
#TBA

# prob (Bonus)
"""
Try to replace the flowers in color or gray-scale sample2.png with sample3.png or other texture you prefer by using the result from (c), and output it as result5.png.
"""
position = np.argwhere(result4_cluster_arr==0)
flower_id = 0
for i in range(1, 4):
    position_new = np.argwhere(result4_cluster_arr==i)
    if position_new.shape[0] > position.shape[0]:
        position = position_new
        flower_id = i

bound_box = np.vstack([ position.min(axis=0), position.max(axis=0)  ])
sample3_rowext_arr = np.concatenate([sample3_arr]*3)
sample3_ext_arr = np.concatenate([sample3_rowext_arr]*3, axis=1)
sample3_extrv_arr = sample3_ext_arr[::-1, :]
#result4_cluster_arr[bound_box[0,0]:bound_box[1,0], bound_box[0,1]:bound_box[1,1]] = sample3_ext_arr[bound_box[0,0]:bound_box[1,0], bound_box[0,1]:bound_box[1,1]]
result5_arr = np.zeros(sample2_arr[:, :, 0].shape)
result5_arr[bound_box[0,0]:bound_box[1,0], bound_box[0,1]:bound_box[1,1]] = sample3_ext_arr[bound_box[0,0]:bound_box[1,0], bound_box[0,1]:bound_box[1,1]]
for i in [x for x in range(4) if x != flower_id]:
    result5_arr = np.where(result4_cluster_arr==i, sample2_arr[:, :, 0], result5_arr)

result5_rv_arr = np.zeros(sample2_arr[:, :, 0].shape)
result5_rv_arr[bound_box[0,0]:bound_box[1,0], bound_box[0,1]:bound_box[1,1]] = sample3_extrv_arr[bound_box[0,0]:bound_box[1,0], bound_box[0,1]:bound_box[1,1]]
for i in [x for x in range(4) if x != flower_id]:
    result5_rv_arr = np.where(result4_cluster_arr==i, sample2_arr[:, :, 0], result5_rv_arr)

utils.save_npArr2JPG(result5_arr, "result5")
utils.save_npArr2JPG(result5_rv_arr, "tmp/result5_rv")

