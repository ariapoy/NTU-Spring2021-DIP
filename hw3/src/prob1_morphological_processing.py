from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

import utils

import pdb

import warnings
warnings.filterwarnings("ignore")

# load image
sample1_arr = utils.load_img2npArr("sample1.png")
print(sample1_arr.shape)

# prob (a)
"""
Perform boundary extraction on sample1.png to extract the objects’ boundaries and output the result as result1.png.
"""
def general_dilation_erosion(img_arr, H, method="erosion"):
    ## Recall of boundary extraction
    """
    F(j, k) - (F(j, k).erosion(H(j, k) ) )
    erosion/dilation:
        1. transform H(j, k) to kernel
        2. Do convolution on F(j, k) and H(j, k)
        3. if sum == np.sum(H(j, k) ):
               erosion
           else:
               dilation
    """
    if np.max(img_arr) == 255:
        F = np.where(img_arr == 255, 1, 0)
    else:
        F = img_arr
    # Step 1 convolution
    T = convolve2d(F, H, boundary='symm', mode='same')
    # Step 2 erosion or dilation
    if method == "erosion":
        G = np.where(T == np.sum(H), 1, 0)
    elif method == "dilation":
        G = np.where(T > 0, 1, 0)
    return G

def bound_extract(img_arr, H):
    if np.max(img_arr) == 255:
        F = np.where(img_arr == 255, 1, 0)
    else:
        F = img_arr
    beta = F - (general_dilation_erosion(F, H, method="erosion"))
    res_arr = utils.int_round(beta*255)
    return res_arr


struct_elem = np.array([
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]
                       ])
result1_arr = bound_extract(sample1_arr, struct_elem)
utils.save_npArr2JPG(result1_arr, "result1")

# prob (b)
"""
Perform hole filling on sample1.png and output the result as result2.png.
"""
def hole_fill(img_arr, H, iters=1):
    if np.max(img_arr) == 255:
        F = np.where(img_arr == 255, 1, 0)
    else:
        F = img_arr
    Fc = 1 - F
    Gi = F
    for i in range(iters):
        Gi = general_dilation_erosion(Gi, H, method="dilation")*Fc
    G = Gi + F
    res_arr = utils.int_round(G*255)
    return res_arr

from collections import deque
def hole_fill_bfs(img_arr):
    if np.max(img_arr) == 255:
        F = np.where(img_arr == 255, 1, 0)
    else:
        F = img_arr
    # [flood-fill algo](https://leetcode.com/problems/flood-fill/discuss/1086688/Python-BFS-easiest-Soln)
    def isValid(image, i, j):
        if i < 0 or i >= image.shape[0] or j < 0 or j >= image.shape[1]:
            return False
        return True
    def floodFill(image, sr, sc, newColor):
        q = deque()
        q.append((sr, sc, image[sr][sc]))
        vis = set()
        row = [-1, 0, 1, 0]
        col = [0, -1, 0, 1]
        while q:
            i, j, k = q.popleft()
            image[i][j] = newColor
            vis.add((i, j))
            for r, c in zip(row, col):
                nRow = i + r
                nCol = j + c
                if isValid(image, nRow, nCol) and (nRow, nCol) not in vis and image[nRow][nCol] == k:
                    image[nRow][nCol] = newColor
                    vis.add((nRow, nCol))
                    q.append((nRow, nCol, k))
        return image
    G = floodFill(F, 0, 0, 2)
    res_arr = utils.int_round( np.where(G==2, 0, 1)*255)
    return res_arr

result2_lec4_arr = hole_fill(sample1_arr, struct_elem)
utils.save_npArr2JPG(result2_lec4_arr, "tmp/result2_lec4")
result2_arr = hole_fill_bfs(sample1_arr)
utils.save_npArr2JPG(result2_arr, "result2")

# prob (c)
"""
Please design an algorithm to count the number of objects in Figure 1. Describe the steps in detail and specify the corresponding parameters.
"""
def conn_comp_label(img_arr, H):
    if np.max(img_arr) == 255:
        F = np.where(img_arr == 255, 1, 0)
    else:
        F = img_arr
    Gi = general_dilation_erosion(F, H, method="dilation")
    G = Gi*F
    G_nonzero = np.where(G==1, -1, 0).copy()
    # Use leader clustering
    m, n = G_nonzero.shape
    label_comp_cnt = 1
    cnt_size = 1
    label_merge_dict = []
    for i in range(1, m-1):
        for j in range(1, n-1):
            if G_nonzero[i, j] == 0:
                continue
            G_sub_arr = G_nonzero[i-cnt_size:i+cnt_size+1, j-cnt_size:j+cnt_size+1]
            if (G_sub_arr > 0).any():
                G_sub_arr_gt0 = G_sub_arr[G_sub_arr > 0]
                if len(np.unique(G_sub_arr_gt0) ) > 1:
                    label_merge_dict.append("{0}to{1}".format(G_sub_arr_gt0.max(), G_sub_arr_gt0.min()))
                G_nonzero[i-cnt_size:i+cnt_size+1, j-cnt_size:j+cnt_size+1] = np.where(G_sub_arr!=0, G_sub_arr.max(), 0)
            elif (G_sub_arr < 0).any():
                G_nonzero[i-cnt_size:i+cnt_size+1, j-cnt_size:j+cnt_size+1] = np.where(G_sub_arr!=0, label_comp_cnt, 0)
                label_comp_cnt += 1
                #label_merge_dict[label_comp_cnt] = {label_comp_cnt}
            else:
                print("I don\'t get it!")
    label_merge_dict = [x.split("to") for x in set(label_merge_dict) ]
    label_merge_dict = [[int(x[0]), int(x[1])] for x in label_merge_dict]
    label_merge_dict = sorted(label_merge_dict)[::-1]
    for cid in label_merge_dict:
        G_nonzero = np.where(G_nonzero==cid[0], cid[1], G_nonzero)
    G_nonzero += 10
    res_arr = utils.int_round(G*255)
    return res_arr, G_nonzero

def conn_comp_label_dict(img_arr, H):
    if np.max(img_arr) == 255:
        F = np.where(img_arr == 255, 1, 0)
    else:
        F = img_arr
    Gi = general_dilation_erosion(F, H, method="dilation")
    G = Gi*F
    G_nonzero = np.where(G==1, -1, 0).copy()
    # Use leader clustering
    m, n = G_nonzero.shape
    label_comp_cnt = 1
    cnt_size = 1
    label_merge_dict = {}
    for i in range(1, m-1):
        for j in range(1, n-1):
            if G_nonzero[i, j] == 0:
                continue
            G_sub_arr = G_nonzero[i-cnt_size:i+cnt_size+1, j-cnt_size:j+cnt_size+1]
            if (G_sub_arr > 0).any():
                G_sub_arr_gt0 = G_sub_arr[G_sub_arr > 0]
                if len(np.unique(G_sub_arr_gt0) ) > 1:
                    label_merge_dict[G_sub_arr_gt0.min()] |= set(np.unique(G_sub_arr_gt0).tolist())
                G_nonzero[i-cnt_size:i+cnt_size+1, j-cnt_size:j+cnt_size+1] = np.where(G_sub_arr!=0, G_sub_arr.max(), 0)
            elif (G_sub_arr < 0).any():
                G_nonzero[i-cnt_size:i+cnt_size+1, j-cnt_size:j+cnt_size+1] = np.where(G_sub_arr!=0, label_comp_cnt, 0)
                label_merge_dict[label_comp_cnt] = {label_comp_cnt}
                label_comp_cnt += 1
            else:
                print("I don\'t get it!")
    label_occur = np.zeros( (max(label_merge_dict), max(label_merge_dict) ) )
    for cid in sorted(label_merge_dict)[::-1]:
        for vid in sorted(label_merge_dict[cid]):
            label_occur[cid-1, vid-1] += 1
            label_occur[vid-1, cid-1] += 1
            #G_nonzero = np.where(G_nonzero==vid, max(label_merge_dict[cid]), G_nonzero)
        #label_occur.append(cid)
    # https://math.stackexchange.com/questions/864604/checking-connectivity-of-adjacency-matrix
    from numpy.linalg import matrix_power
    label_occur = np.sign(matrix_power(np.sign(label_occur), label_occur.shape[0]))
    for row in label_occur:
        for col in row.nonzero()[0]:
            G_nonzero = np.where(G_nonzero==col+1, row.argmax()+1, G_nonzero)
    res_arr = utils.int_round(G*255)
    return res_arr, G_nonzero

prob1c_hole_fill_arr = hole_fill_bfs(sample1_arr)
prob1c_arr, label_comp = conn_comp_label_dict(prob1c_hole_fill_arr, struct_elem)
utils.save_npArr2JPG(prob1c_arr, "tmp/prob1c")
print(len(np.unique(label_comp))-1)
fig, ax = plt.subplots()
label_comp_fig = np.ma.masked_where(label_comp == 0, label_comp)
cmap = plt.cm.RdBu
cmap.set_bad(color='black')
ax.matshow(label_comp_fig, interpolation='none', cmap=cmap)
fig.savefig("tmp/prob1c_cluster.png")

