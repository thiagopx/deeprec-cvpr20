import cv2
import numpy as np
from skimage import measure, morphology
from scipy import stats
from docrec.ndarray.utils import first_nonzero


def binary_index(image, mask=None):
    
    assert image.ndim == 2
    
    h, w = image.shape
    if mask is None:
        mask = 255 * np.ones((h, w), dtype=np.uint8)

    counts = np.bincount(image[mask == 255])#, minlength=256)
    s = first_nonzero(counts)
    probs = counts[s :] / float(counts.sum())
    n = probs.size
    r = np.arange(n)
    m = n / 2.0
    a = (m ** -2)
    b_index = np.sum(probs * (a * (r - m) ** 2))

    return b_index


def entropy(img, mask=None):
    ''' Shannon entropy. '''
    
    assert img.ndim == 2
    
    if mask is None:
        mask = np.ones_like(img).astype(np.bool)
    else:
        mask = (mask == 255)
    
    hist = np.bincount(img[mask].flatten(), minlength=256)
    probs = hist / float(hist.sum())
    
    return stats.entropy(probs)
    

def median_complexity(img, mask=None, scale=3):
    
    assert img.ndim == 2
    
    is_binary = all(val in [0, 255] for val in np.unique(img))
    assert is_binary
    
    if mask is None:
        mask = np.ones_like(img).astype(np.bool)
    else:
        mask = (mask == 255)
    
    # B/W stats
    n = mask.sum()
    nw = np.logical_and(img == 255, mask).sum()   # original white pixels
    nb = n - nw                                   # original black pixels
    
    diff = img.astype(np.int16) - cv2.medianBlur(img, scale)
    wb = np.logical_and(diff == 255, mask).sum()  # white to black
    bw = np.logical_and(diff == -255, mask).sum() # black to white
        
    if nw == 0:
        if nb == 0:
            return np.inf
        return float(bw) / nb
    elif nb == 0:
        return float(wb) / nw
    return max(float(wb) / nw, float(bw) / nb)

# https://stats.stackexchange.com/questions/17109/measuring-entropy-information-patterns-of-a-2d-binary-matrix

def changes_complexity(img):
    '''
    Binary image complexity
    '''
    assert img.ndim == 2
   
    is_binary = all(val in [0, 255] for val in np.unique(img))
    assert is_binary
  
    img_ = img.astype(np.bool)
    h, w = img_.shape
    k = np.diff(img_, axis=0).sum() + np.diff(img_, axis=1).sum()
  
    return float(k) / ((h - 1) * w + (w - 1) * h)

def psnr_complexity(img, scale=3):
    
    assert img.ndim == 2
    
    is_binary = all(val in [0, 255] for val in np.unique(img))
    assert is_binary
    
    # B/W stats
    n = img.size
    nw = (img == 255).sum()   # original white pixels
    nb = n - nw               # original black pixels
    
    inv = 255 - img
    blurred  = cv2.medianBlur(img, scale)
    blurred_inv  = cv2.medianBlur(inv, scale)
    psnr = measure.compare_psnr(img, blurred, 255)
    psnr_inv = measure.compare_psnr(inv, blurred_inv, 255)
    
    return min(psnr, psnr_inv)
    

# To improve
def porosity(img, scale=3):
    
    assert img.ndim == 2
    
    is_binary = all(val in [0, 255] for val in np.unique(img))
    assert is_binary
    
    # B/W stats
    n = img.size
    nw = (img == 255).sum()   # original white pixels
    nb = n - nw               # original black pixels
    
    # Area of holes
    img_ = img.astype(np.bool)
    inv = np.logical_not(img_)
    ab_h = np.logical_xor(
        morphology.remove_small_holes(img_, scale), img_
    ).sum()
    aw_h = np.logical_xor(
        morphology.remove_small_holes(inv, scale), inv
    ).sum()
        
    return max(float(ab_h) / nb, float(aw_h) / nw)
