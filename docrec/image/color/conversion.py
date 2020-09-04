import numpy as np
import cv2

def rgb2yiq(rgb):
    
    r, g, b = cv2.split(rgb)
    y = 0.299 * r +  0.587 * g +  0.114 * b
    i = 0.596 * r + -0.274 * g + -0.322 * b
    q = 0.211 * r + -0.523 * g +  0.312 * b
    return cv2.merge((y, i, q))


def bgr2yiq(bgr):
    
    b, g, r = cv2.split(bgr)
    rgb = cv2.merge((r, g, b))
    return rgb2yiq(rgb)

def normalize(image, epsilon=1e-12):
    
    ''' Normalize n-channels image to obtain same variance. '''
    normalized = np.empty(image.shape, dtype=np.float32)
    for c in range(image.shape[2]):
        channel = image[:, :, c]
        normalized[:, :, c] = (channel - channel.mean()) / \
            (channel.std() + epsilon)
    return normalized