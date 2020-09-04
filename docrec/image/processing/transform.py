import numpy as np
import cv2
from skimage import measure


def distance(image, metric=cv2.NORM_L2, window=None):
    ''' Compute the distance transform. '''

    # align center with centroid
    h, w = image.shape
    moments = measure.moments(image, order=1)
    yc = int(moments[1, 0] / moments[0, 0])
    xc = int(moments[0, 1] / moments[0, 0])
    # distance computation
    wp, hp = window if window is not None else (w, h)
    xb, yb = wp // 2 - xc, hp // 2 - yc
    base = np.zeros((hp, wp), dtype=np.uint8)
    base[yb : yb + h, xb : xb + w] = image
    result = cv2.distanceTransform(255 - base, metric, cv2.DIST_MASK_PRECISE)
    return result, base