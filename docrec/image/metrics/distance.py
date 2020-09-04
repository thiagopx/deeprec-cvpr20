import numpy as np
import cv2
from skimage import measure


def modified_hausdorff(image1, image2, metric=cv2.NORM_L2):
    ''' Compute the Modified Hausdorff distance. '''
    
    # Align center with centroids
    h1, w1 = image1.shape
    h2, w2 = image2.shape
    m1 = measure.moments(image1, order=1)
    m2 = measure.moments(image2, order=1)
    xc1, yc1 = int(m1[1, 0] / m1[0, 0]), int(m1[0, 1] / m1[0, 0])
    xc2, yc2 = int(m2[1, 0] / m2[0, 0]), int(m2[0, 1] / m2[0, 0])
    dx1, dy1 = (xc1 - w1 / 2), (yc1 - h1 / 2)
    dx2, dy2 = (xc2 - w2 / 2), (yc2 - h2 / 2)
    
    # Contour extraction
    _, contours1, hierarchy1 =  cv2.findContours(
        image1.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    _, contours2, hierarchy2 =  cv2.findContours(
        image2.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Contours drawing
    padded1 = np.zeros_like(image1)
    padded2 = np.zeros_like(image2)
    idx = 0
    while idx >= 0:
        padded1 = cv2.drawContours(padded1, contours1, idx, (255, 255, 255))
        idx = hierarchy1[0][idx][0]
    idx = 0
    while idx >= 0:
        padded2 = cv2.drawContours(padded2, contours2, idx, (255, 255, 255))
        idx = hierarchy2[0][idx][0]
    
    # Padding
    padded1 = np.pad(
        padded1, ((max(0, -dy1), max(0, dy1)), (max(0, -dx1), max(0, dx1))),
        mode='constant', constant_values=0
    )
    padded2 = np.pad(
        padded2, ((max(0, -dy2), max(0, dy2)), (max(0, -dx2), max(0, dx2))),
        mode='constant', constant_values=0
    )
    
    # Distance computations
    h1, w1 = padded1.shape
    h2, w2 = padded2.shape
    h, w = max(h1, h2) + 2, max(w1, w2) + 2
    dx1, dy1 = (w - w1) / 2, (h - h1) / 2
    dx2, dy2 = (w - w2) / 2, (h - h2) / 2
    base1 = np.zeros((h, w), dtype=np.uint8)
    base2 = np.zeros((h, w), dtype=np.uint8)
    base1[dy1 : dy1 + h1, dx1 : dx1 + w1] =  padded1
    base2[dy2 : dy2 + h2, dx2 : dx2 + w2] =  padded2
    dist1 = cv2.distanceTransform(255 - base1, metric, cv2.DIST_MASK_PRECISE)
    dist2 = cv2.distanceTransform(255 - base2, metric, cv2.DIST_MASK_PRECISE)
    h12 = dist1[base2 == 255].mean()
    h21 = dist2[base1 == 255].mean()

    return max(h12, h21)