from PIL import Image
import numpy as np
import numpy.linalg as la
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

import utils

import pdb

# load image
sample3_arr = utils.load_img2npArr("sample3.jpg")
sample5_arr = utils.load_img2npArr("sample5.jpg")

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

# If homogenous that means we expand our matrix to make shift as linear transformation
def get_grid(x, y, homogenous=False):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1]))) if homogenous else coords

def linear_geo_transform(img_arr, angle=-78, tx=-200, ty=150, sx=1.5, sy=1.5, operation="backward"):
    height, width = img_arr.shape
    # As we conduct geometrical transformation at the center of image
    R = rotate(angle)
    T = translate(tx, ty)
    S = scale(sx, sy)
    # Lec 3 page 56
    # Forward, it is not good approach.
    A = S @ T @ R
    # Backward, it is better appraoch.
    Ainv = np.linalg.inv(A)
    if operation == "backward":
        # Generate grip map
        xycoords = get_grid(width, height, True)
        # Shift to center
        xycoords[0], xycoords[1] = xycoords[0] - width/2, xycoords[1] - height/2
        # Apply inverse transform
        uvcoords = Ainv @ xycoords
        # Init mapping pixels and shift to original position
        u, v = uvcoords[0] + width/2, uvcoords[1] + height/2
        x, y = xycoords[0] + width/2, xycoords[1] + height/2
        # Get pixels within image boundaries
        indices = np.where((u >= 0) & (u < width) &
                           (v >= 0) & (v < height)
                          )
        upix, vpix = u[indices], v[indices]
        xpix, ypix = x[indices], y[indices]
    elif operation == "forward":
        uvcoords = get_grid(width, height, True)
        uvcoords[0] = uvcoords[0] - width/2
        uvcoords[1] = uvcoords[1] - height/2
        xycoords = S @ T @ R @ uvcoords
        u, v = uvcoords[0] + width/2, uvcoords[1] + height/2
        x, y = xycoords[0] + width/2, xycoords[1] + height/2
        indices = np.where((x >= 0) & (x < width) &
                           (y >= 0) & (y < height)
                          )
        upix, vpix = u[indices], v[indices]
        xpix, ypix = x[indices], y[indices]
    # Draw the new canvas array
    canvas = np.zeros_like(img_arr)
    # Draw the original image to new image
    # Wiht nearest neighbor (round) interpolation method.
    canvas[ypix.astype(int), xpix.astype(int)] = img_arr[vpix.astype(int), upix.astype(int)]
    return canvas

result6_arr = linear_geo_transform(sample3_arr)
result6_forward_arr = linear_geo_transform(sample3_arr, operation="forward")
utils.save_npArr2JPG(result6_arr, "result6")
utils.save_npArr2JPG(result6_forward_arr, "tmp/result6_forward")

# prob (b)
"""
Imagine that there is a black hole in the center absorbing sample5.jpg. Please design a method to make sample5.jpg look like sample6.jpg as much as possible and save the output as result7.jpg.
"""

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def spiral_geo_transform(img_arr, k=200):
    height, width = img_arr.shape
    xycoords = get_grid(width, height)
    xycoords[0], xycoords[1] = xycoords[0] - width/2, xycoords[1] - height/2
    # Cart axis to Polar axis of xy
    rthetacoords = np.array( cart2pol(xycoords[0], xycoords[1])  )
    # Normalization of radius
    r_normalization = np.sqrt( la.norm(rthetacoords[0])  )
    r_max = rthetacoords[0].max()
    # spiral transformation
    rhophicoords = np.zeros(rthetacoords.shape)
    rhophicoords[1] = rthetacoords[1] + k/(rthetacoords[0] + 0.0001)
    rhophicoords[0] = rthetacoords[0]
    # Polar axis to Cart axis of uv
    uvcoords = np.array( pol2cart(rhophicoords[0], rhophicoords[1])  )
    u, v = uvcoords[0] + width/2, uvcoords[1] + height/2
    x, y = xycoords[0] + width/2, xycoords[1] + height/2
    indices = np.where((u >= 0) & (u < width) &
                       (v >= 0) & (v < height))
    upix, vpix = u[indices], v[indices]
    xpix, ypix = x[indices], y[indices]
    canvas = np.zeros_like(img_arr)
    canvas[ypix.astype(int), xpix.astype(int)] = img_arr[vpix.astype(int), upix.astype(int)]
    return canvas

result7_arr = spiral_geo_transform(sample5_arr, k=200)
utils.save_npArr2JPG(result7_arr, "result7")

