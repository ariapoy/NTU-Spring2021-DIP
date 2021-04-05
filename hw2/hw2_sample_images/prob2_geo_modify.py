from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

import utils

import pdb

# load image
sample3_arr = utils.load_img2npArr("sample3.jpg")

# prob (a)
"""
Please design an algorithm to make sample3.jpg become sample4.jpg. Output the result as result6.jpg. Please describe your method and implementation details clearly. (hint: you may perform rotation, scaling, translation, etc.)
Reference: [Image Geometric Transformation In Numpy and OpenCV](https://towardsdatascience.com/image-geometric-transformation-in-numpy-and-opencv-936f5cd1d315)
"""

def rotate(angle):
    angle = np.radians(angle)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return R

def translate(tx, ty):
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    return T

def scale(sx, sy):
    S = np.array([
    [sx, 0, 0],
    [0, sy, 0],
    [0, 0, 1]
    ])
    return S

def get_grid(x, y, homogenous=False):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1]))) if homogenous else coords

def prob2b_geo_modify(img_arr, angle=-78, tx=-200, ty=150, sx=1.5, sy=1.5, operation="backward"):
    height, width = img_arr.shape
    # As we conduct geometrical transformation at the center of image
    Torig = translate(width/2, height/2)
    R = rotate(angle)
    T = translate(tx, ty)
    S = scale(sx, sy)
    # Lec 3 page 56
    # Forward, it is not good approach.
    A = Torig @ S @ T @ R @ np.linalg.inv(Torig)
    # Backward, it is better appraoch.
    Ainv = np.linalg.inv(A)
    if operation == "backward":
        uvcoords = get_grid(width, height, True)
        u, v = uvcoords[0], uvcoords[1]
        # Apply inverse transform and round it (nearest neighbour interpolation)
        xycoords = (Ainv@uvcoords).astype(np.int)
        x, y = xycoords[0, :], xycoords[1, :]
        # Get pixels within image boundaries
        indices = np.where((x >= 0) & (x < width) &
                           (y >= 0) & (y < height)
                          )
        upix, vpix = u[indices], v[indices]
        xpix, ypix = x[indices], y[indices]
    elif operation == "forward":
        pass
    # Draw the new canvas array
    canvas = np.zeros_like(img_arr)
    canvas[vpix.astype(int), upix.astype(int)] = img_arr[ypix.astype(int),xpix.astype(int)]
    return canvas

result6_arr = prob2b_geo_modify(sample3_arr)
utils.save_npArr2JPG(result6_arr, "result6")

