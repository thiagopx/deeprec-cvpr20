# Adapted from http://stackoverflow.com/questions/23506105/extracting-text-opencv
import cv2
from skimage import morphology, measure
import numpy as np
import matplotlib.pyplot as plt

def remove_lines(img, x_factor=0.3, max_lines=50, threshold=0.45):
    ''' Remove vertical and horizontal lines from strips. '''
    
    # Dimensions
    h, w = img.shape
    
    # Resulting image
    res = img.copy()
    
    # Threshold
    th, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_bw = 255 - bw
    
    # Detect vertical markers
    dy = int(h /max_lines)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dy))
    v_lines_markers = cv2.morphologyEx(inv_bw, cv2.MORPH_OPEN, v_kernel)

    # Detect horizontal markers
    dx = int(x_factor * w)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dx, 1))
    h_lines_markers = cv2.morphologyEx(inv_bw, cv2.MORPH_OPEN, h_kernel)
    
    # Removing process
    mask = cv2.bitwise_or(v_lines_markers, h_lines_markers).astype(np.bool)
    n_labels, labels = cv2.connectedComponents(inv_bw)
    lines_labels = np.unique(labels[mask])
    for label in lines_labels:
        res[labels == label] = 255
        
    return res