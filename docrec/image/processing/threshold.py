import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

# Deprecated
def otsu(image):
   assert image.ndim == 2
            
   return image >= threshold_otsu(image)

def threshold_hasan_karan(image, min_contrast=30, mask=None):
    ''' Nonhistogram-based theshold. 

    Morphological Text Extraction from Images
    Yassin M. Y. Hasan and Lina J. Karan
    IEEE TRANSACTIONS ON IMAGE PROCESSING, 2000
        
    '''
    assert image.ndim == 2 # grayscale
    
    if mask is None:
        mask = 255 * np.ones_like(image)
    
    # Check if the image is already binary
    is_binary = all(val in [0, 255] for val in np.unique(image))
    if is_binary:
        return 127, image
    
    # Check low contrast
    if (image.max() - image.min()) < min_contrast:
        return -1, None
    
    g1 = np.array([[ 0, 0, 0],
                   [-1, 0, 1],
                   [ 0, 0, 0]])
    g2 = g1.T
    s1 = cv2.filter2D(image, -1, g1)
    s2 = cv2.filter2D(image, -1, g2)
    s = cv2.max(np.abs(s1), np.abs(s2)).astype(np.int64)
    total_s = s.sum()
    if total_s == 0:
        return -1, None
    
    gamma = int(float((image * s).sum()) / total_s)

    return cv2.threshold(image, gamma, 255, cv2.THRESH_BINARY)
    
    