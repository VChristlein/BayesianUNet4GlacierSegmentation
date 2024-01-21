import cv2
import numpy as np
from scipy.ndimage.filters import convolve

def kuan(img, window_size=7, looks=2, eps=10e-6):
    mean_kernel = np.ones((window_size, window_size)) / window_size**2
    mean = convolve(img, mean_kernel)
    var = convolve((img - mean)**2, mean_kernel)
    ci_squared_inv =  mean**2 / (var + eps)
    cu_squared = 1/looks
    W = (1 - cu_squared * ci_squared_inv)/ (1 + cu_squared)

    R = img * W + mean * (1 - W)
    return R.astype(img.dtype)

def enhanced_lee(img, window_size=7, looks=2, damp_factor=1, eps=10e-6):
    mean_kernel = np.ones((window_size, window_size)) / window_size**2
    mean = convolve(img, mean_kernel)
    std = np.sqrt(convolve((img - mean)**2, mean_kernel))
    cu = np.sqrt(1/looks)
    Cmax = np.sqrt(1 + 2/looks)
    ci = std / (mean + eps)
    W = np.exp(-damp_factor * (ci - cu) / (Cmax - ci))
    R = mean + W * (img - mean)          # heterogeneous region  -> MMSE filter
    R[ci <= cu] = mean[ci <= cu]        # homogenous region => mean
    R[ci >= Cmax] = img[ci >= Cmax]     # no noise

    return R.astype(img.dtype)


def get_denoise_filter(denoise, denoise_parameters=None):
    """
    Return preconfigured Denoise Filter function with only image as input
    :param denoise: Name of the filter
    :param denoise_parameters: Filter parameters
    :return: filter_func(img)
    """
    denoise = denoise.lower()
    if denoise == 'bilateral':
        if denoise_parameters:
            return lambda  img:cv2.bilateralFilter(img,**denoise_parameters)
        else:
            return lambda img:cv2.bilateralFilter(img, 20, 80, 80)
    elif denoise == 'median':
        if denoise_parameters:
            return lambda img: cv2.medianBlur(img, **denoise_parameters)
        else:
            return lambda img: cv2.medianBlur(img, 5)
    elif denoise == 'nlmeans':
        if denoise_parameters:
            return lambda img: cv2.fastNlMeansDenoising(img,**denoise_parameters)
        else:
            return lambda img: cv2.fastNlMeansDenoising(img)
    elif denoise == 'kuan':
        return lambda img: kuan(img)
    elif denoise == 'enhanced_lee':
        return lambda img: enhanced_lee(img)
    elif denoise == 'none':
        return None

    raise NotImplementedError(denoise + " is not Implemented")






