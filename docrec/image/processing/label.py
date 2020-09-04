import numpy as np
from numba import jit

def forward_backward_propagation(A):
    ''' Extract connected components. Implemented from:
    
        Morphological Text Extraction from Images
        Yassin M. Y. Hasan and Lina J. Karan
        IEEE TRANSACTIONS ON IMAGE PROCESSING, 2000
    '''
    
    @jit(nopython=True)
    def _propagate(B):
        
        h, w = B.shape
        
        # Row-wise backpropagation
        for i in xrange(h - 1, 0, -1):
            for j in xrange(w):
                b, t = B[i, j], B[i - 1, j]
                min_val, max_val = (b, t) if b <= t else (t, b)
                if min_val > 0:
                    B[i, j] = max_val
                    B[i - 1, j] = max_val
                    
        # Column-wise backpropagation
        for i in xrange(h):
            for j in xrange(w - 1, 0, -1):
                r, l = B[i, j], B[i, j - 1]
                min_val, max_val = (r, l) if r <= l else (l, r)
                if min_val > 0:
                    B[i, j] = max_val
                    B[i, j - 1] = max_val
        
        # Row-wise forward propagation
        for i in xrange(h - 1):
            for j in xrange(w):
                t, b = B[i, j], B[i + 1, j]
                min_val, max_val = (t, b) if t <= b else (b, t)
                if min_val > 0:
                    B[i, j] = max_val
                    B[i + 1, j] = max_val

        # Column-wise forward propagation
        for i in xrange(h):
            for j in xrange(w - 1):
                l, r = B[i, j], B[i, j + 1]
                min_val, max_val = (l, r) if l <= r else (r, l)
                if min_val > 0:
                    B[i, j] = max_val
                    B[i, j + 1] = max_val
                
    
    B = np.zeros(A.shape, dtype=np.uint64)
    
    # Assign sequential labels
    idx = np.where(A > 0)
    max_label = idx[0].size
    B[idx] = np.arange(1, max_label + 1)
    
    # Label propagation
    B_old = B.copy()
    _propagate(B)
    while (B != B_old).any():
        B_old[:] = B
        _propagate(B)

    # Sequential relabeling
    labels = np.unique(B)
    max_label = 0
    for label in labels:
        B[B == label] = max_label
        max_label += 1
            
    return B
