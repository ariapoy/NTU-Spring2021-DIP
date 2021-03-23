from PIL import Image
import numpy as np

import utils

import pdb

# load image
sample1_arr = utils.load_img2npArr("sample1.jpg")

# prob (a)
def rgb2gray(rgb, r_scale=0.2989, g_scale=0.5870, b_scale=0.1140):
    '''
    formula: Grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
    ref: https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    '''
    return np.dot(rgb[..., :3], [r_scale, g_scale, b_scale])

result1_arr = rgb2gray(sample1_arr)
result1 = Image.fromarray(result1_arr)
utils.save_npArr2JPG(result1_arr, "1_result")

# prob (b)
result2_arr = sample1_arr[:, ::-1, :]
utils.save_npArr2JPG(result2_arr, "2_result")

