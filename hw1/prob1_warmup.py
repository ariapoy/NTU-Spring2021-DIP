from PIL import Image
import numpy as np

import pdb

# prob (a)
sample1 = Image.open("sample1.jpg")
sample1_arr = np.array(sample1)

def rgb2gray(rgb, r_scale=0.2989, g_scale=0.5870, b_scale=0.1140):
    '''
    formula: Grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
    ref: https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    '''
    return np.dot(rgb[..., :3], [r_scale, g_scale, b_scale])

result1_arr = rgb2gray(sample1_arr)
result1 = Image.fromarray(result1_arr)

'''
Question 1
Can I use `convert('L')`?
'''
# convert('L'): JPEG images do not support the alpha(transparency) channel
result1.convert('L').save("1_result.jpg", "JPEG")

# prob (b)
result2_arr = sample1_arr[:, ::-1, :]
result2 = Image.fromarray(result2_arr)
result2.save("2_result.jpg", "JPEG")

