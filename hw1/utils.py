from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pdb

def load_img2npArr(name):
    img = Image.open(name)
    img_arr = np.array(img)
    return img_arr

def save_npArr2JPG(img_arr, name):
    img = Image.fromarray(img_arr)
    img.convert("RGB").save("{0}.jpg".format(name), "JPEG")
    print("Save fig {0}.jpg!".format(name))

def int_round(img_arr):
    return np.array(np.round(img_arr), dtype=np.uint8)

def plot_hist(name, *args):
    hist = []
    L = 256
    for h in args:
        hist.append(h)
    n_plots = len(hist)
    fig, axes = plt.subplots(1, n_plots)
    fig.set_size_inches(16, 6)
    fig.suptitle("Compare histogram of {0}".format(name))
    ax_bin = np.diff(np.arange(L + 1))
    for i in range(n_plots):
        # hist[i] = [intensity, cnt]
        axes[i].bar(hist[i][0], hist[i][1], width=ax_bin, align="edge")
    plt.savefig( "{0}.png".format(name) )
    plt.clf()

def plot_transfer(name, T, prefix_dir="."):
    r = np.linspace(0., 1.0, 100)
    s = T(r)
    plt.plot(r, s)
    plt.title( "{0}".format(name) )
    plt.savefig( "{1}/prob2g_transfer_{0}.png".format(name, prefix_dir) )
    plt.clf()

def plot_spectrum(im_fft, fig_name):
    # A logarithmic colormap
    plt.figure()
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.title('Fourier transform')
    plt.savefig("{0}.png".format(fig_name))
    plt.clf()

def PSNR(F_true, F):
    MSE = np.sum((F_true - F)**2) / np.prod(F.shape)
    res = 10 * np.log10(255**2 / MSE)
    return res

def plot_param_search(f, param_name, param_grid, img_orig, img_noise, img_name):
    '''
    Input:
    f: [object] function
    param_name: [str]
    param_grid: [list]
    '''
    metric = [] # y-axis
    for p in param_grid:
        params = {param_name: p}
        img_recover = f(img_noise, **params)
        score = PSNR(img_orig, img_recover)
        metric.append(score)
        
    plt.figure()
    plt.plot(param_grid, metric)
    plt.title('Different {0} for PSNR'.format(param_name))
    plt.savefig( "{0}_param{1}.png".format(img_name, param_name) )
    plt.clf()

