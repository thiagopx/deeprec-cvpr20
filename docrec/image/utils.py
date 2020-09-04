import numpy as np
import os
from ..ndarray.utils import first_nonzero, last_nonzero


def join(image1, image2, offset1=0, offset2=0, cval=0):
   ''' Join images horizontally considering offset. '''

   h1, w1 = image1.shape[: 2]
   h2, w2 = image2.shape[: 2]

   # support image
   h = max(h1 + offset1, h2 + offset2)
   w = w1 + w2
   dim = (h, w, image1.shape[2]) if image1.ndim == 3 else (h, w)
   support = np.empty(dim, dtype=image1.dtype)
   support[:] = cval

   # joining
   support[offset1 : offset1 + h1, : w1] = image1
   support[offset2 : offset2 + h2, w1 :] = image2
   return support

def center(image, window_size=(20, 20), cval=0):
    ''' Center image in a window. '''

    hs, ws = window_size
    # image
    image = image[: hs, : ws]     # fix dimensions
    h, w = image.shape[: 2]
    y, x = int(h / 2), int(w / 2) # center

    # support image
    ys, xs = int(hs / 2), int(ws / 2) # center
    dim = (hs, ws, image.shape[2]) if image.ndim == 3 else (hs, ws)
    support = np.zeros(dim, dtype=np.uint8)
    support[:] = cval

    # centering
    dy, dx = ys - y, xs - x # displacement (image -> support)
    support[dy : dy + h, dx : dx + w] = image
    return support
