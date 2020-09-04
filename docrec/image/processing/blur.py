import cv2
import numpy as np


def blur_hasan_karan(img):
    ''' Extract connected components. Implemented from:
    
        Morphological Text Extraction from Images
        Yassin M. Y. Hasan and Lina J. Karan
        IEEE TRANSACTIONS ON IMAGE PROCESSING, 2000
    '''
    
    assert img.ndim == 2 # grayscale
    
    is_binary = all(val in [0, 255] for val in np.unique(img))
    if is_binary:
        return img        
    
    s_33 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_oc = cv2.morphologyEx(img, cv2.MORPH_OPEN, s_33)
#    img_oc = cv2.morphologyEx(img_oc, cv2.MORPH_CLOSE, s_33)
    img_co = cv2.morphologyEx(img, cv2.MORPH_CLOSE, s_33)
#    img_co = cv2.morphologyEx(img_co, cv2.MORPH_OPEN, s_33)

    # Mean
    blur = cv2.multiply(
        img_oc.astype(np.int16) +  img_co.astype(np.int16),
        0.5
    ).astype(np.uint8)
        
    return blur
    