import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def manual_linear_contrast(image):
    """Линейное контрастирование"""
    img_float = image.astype(np.float32)
    
    min_val = np.min(img_float, axis=(0, 1), keepdims=True)
    max_val = np.max(img_float, axis=(0, 1), keepdims=True)
    
    diff = max_val - min_val
    diff[diff == 0] = 1 
    
    contrasted = (img_float - min_val) * (255.0 / diff)
    
    return np.clip(contrasted, 0, 255).astype(np.uint8)

def manual_gamma_correction(image, gamma):
    """Гамма-коррекция"""
    img_float = image.astype(np.float32) / 255.0
    corrected = np.power(img_float, 1.0 / gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)

def rgb_to_gray(image):
    """Ручной перевод RGB в Grayscale по стандарту яркости"""
    if len(image.shape) == 3:
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        return gray
    return image.astype(np.float64)


def manual_adaptive_mean(image, block_size, C):
    """Адаптивное среднее"""
    gray = rgb_to_gray(image)
    h, w = gray.shape
    pad = block_size // 2
    
    padded = np.pad(gray, pad, mode='edge')
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    
    br = integral[block_size-1 : h+block_size-1, block_size-1 : w+block_size-1]
    
    tr = np.zeros((h, w))
    bl = np.zeros((h, w))
    tl = np.zeros((h, w))
    
    if pad > 0:
        tr = integral[0 : h, block_size-1 : w+block_size-1]
        bl = integral[block_size-1 : h+block_size-1, 0 : w]
        tl = integral[0 : h, 0 : w]
        
    local_sum = br - tr - bl + tl
    local_mean = local_sum / (block_size ** 2)
    
    out = np.where(gray > (local_mean - C), 255, 0)
    return out.astype(np.uint8)

def manual_adaptive_gaussian(image, block_size, C):
    """
    Адаптивный Гаусс. 
    Оптимизировано с помощью сепарабельной свертки.
    """
    gray = rgb_to_gray(image)
    h, w = gray.shape
    pad = block_size // 2
    
    sigma = 0.3 * ((block_size - 1) * 0.5 - 1) + 0.8
    ax = np.linspace(-(block_size - 1) / 2., (block_size - 1) / 2., block_size)
    kernel_1d = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    
    padded_h = np.pad(gray, ((0, 0), (pad, pad)), mode='edge')
    windows_h = sliding_window_view(padded_h, (1, block_size))
    filtered_h = np.sum(windows_h.reshape(h, w, block_size) * kernel_1d, axis=2)
    
    padded_v = np.pad(filtered_h, ((pad, pad), (0, 0)), mode='edge')
    windows_v = sliding_window_view(padded_v, (block_size, 1))
    local_mean = np.sum(windows_v.reshape(h, w, block_size) * kernel_1d, axis=2)
    
    out = np.where(gray > (local_mean - C), 255, 0)
    return out.astype(np.uint8)