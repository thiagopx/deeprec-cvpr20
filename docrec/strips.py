import os
import re
import numpy as np
import cv2
import random
import copy

# from .strip import Strip
from .ndarray.utils import first_nonzero, last_nonzero


class Strip(object):
    ''' Strip image.'''

    def __init__(self, image, index, mask=None):

        h, w = image.shape[: 2]
        if mask is None:
            mask = 255 * np.ones((h, w), dtype=np.uint8)

        self.h = h
        self.w = w
        self.image = cv2.bitwise_and(image, image, mask=mask)
        self.index = index
        self.mask = mask

        self.offsets_l = np.apply_along_axis(first_nonzero, 1, self.mask) # left border (hor.) offsets
        self.offsets_r = np.apply_along_axis(last_nonzero, 1, self.mask)   # right border (hor.) offsets
        self.approx_width = int(np.mean(self.offsets_r - self.offsets_l + 1))


    def copy(self):
        ''' Copy object. '''

        return copy.deepcopy(self)


    def shift(self, disp):
        ''' shift strip vertically. '''

        M = np.float32([[1, 0, 0], [0, 1, disp]])
        self.image = cv2.warpAffine(self.image, M, (self.w, self.h))
        self.mask = cv2.warpAffine(self.mask, M, (self.w, self.h))
        self.offsets_l = np.apply_along_axis(first_nonzero, 1, self.mask) # left border (hor.) offsets
        self.offsets_r = np.apply_along_axis(last_nonzero, 1, self.mask)   # right border (hor.) offsets
        self.approx_width = int(np.mean(self.offsets_r - self.offsets_l + 1))
        return self


    def filled_image(self):
        ''' Return image with masked-out areas in white. '''

        return cv2.bitwise_or(
            self.image, cv2.cvtColor(
                cv2.bitwise_not(self.mask), cv2.COLOR_GRAY2RGB
            )
        )


    def is_blank(self, blank_tresh=127):
        ''' Check whether is a blank strip. '''

        blurred = cv2.GaussianBlur(
            cv2.cvtColor(self.filled_image(), cv2.COLOR_RGB2GRAY), (5, 5), 0
        )
        return (blurred < blank_tresh).sum() == 0


    def stack(self, other, disp=0, filled=False):
        ''' Stack horizontally with other strip. '''

        y1_min, y1_max = 0, self.h - 1
        y2_min, y2_max = 0, other.h - 1
        y_inter_min = max(0, disp)
        y_inter_max = min(y1_max, y2_max + disp) + 1
        h_inter = y_inter_max - y_inter_min

        # borders coordinates
        r1 = self.offsets_r[y_inter_min : y_inter_max]

        if disp >= 0:
            l2 = other.offsets_l[: h_inter]
        else:
            l2 = other.offsets_l[-disp : -disp + h_inter]

        # horizontal offset
        offset = self.w - np.min(l2 + self.w - r1) + 1

        # union
        y_union_min = min(0, disp)
        y_union_max = max(y1_max, y2_max + disp) + 1
        h_union = y_union_max - y_union_min

        min_h, max_h = min(self.h, other.h), max(self.h, other.h)

        # new image / mask
        temp_image = np.zeros((h_union, offset + other.w, 3), dtype=np.uint8)
        temp_mask = np.zeros((h_union, offset + other.w), dtype=np.uint8)
        if disp >= 0:
            temp_image[: self.h, : self.w] = self.image
            temp_image[disp : disp + other.h, offset :] += other.image
            temp_mask[: self.h, : self.w] = self.mask
            temp_mask[disp : disp + other.h, offset :] += other.mask
        else:
            temp_image[-disp : -disp + self.h, : self.w] = self.image
            temp_image[: other.h, offset :] += other.image
            temp_mask[-disp : -disp + self.h, : self.w] = self.mask
            temp_mask[: other.h, offset :] += other.mask

        self.h, self.w = temp_mask.shape
        self.image = temp_image
        self.mask = temp_mask
        self.offsets_l =np.apply_along_axis(first_nonzero, 1, self.mask)
        self.offsets_r =np.apply_along_axis(last_nonzero, 1, self.mask)
        if filled:
            self.image = self.filled_image()
        return self


    def crop_vertically(self, y1, y2):
        ''' Crop the strip vertically from h1 to h2. '''

        self.offsets_l = self.offsets_l[y1 : y2 + 1]
        self.offsets_r = self.offsets_r[y1 : y2 + 1]
        x1 = self.offsets_l.min()
        x2 = self.offsets_r.max()
        self.offsets_l -= x1
        self.offsets_r -= x1
        self.image = self.image[y1 : y2 + 1, x1 : x2 + 1] # height can be different from y2 - y1 for the bottom part of the document


class Strips(object):
    ''' Strips operations manager.'''

    def __init__(self, path=None, filter_blanks=True, blank_tresh=127):
        ''' Strips constructor.

        @path: path to a directory containing strips (in case of load real strips)
        @strips_list: list of strips (objects of Strip class)
        @filter_blanks: true-or-false flag indicating the removal of blank strips
        @blank_thresh: threshold used in the blank strips filtering
        '''

        self.left_extremities = []
        self.right_extremities = []
        self.strips = []

        # self.artificial_mask = False
        if path is not None:
            assert os.path.exists(path)

            # load the strips
            self._load_data(path)

            # remove low content ('blank') strips
            if filter_blanks:
                self.strips = [strip for strip in self.strips if not strip.is_blank(blank_tresh)]
                new_indices = np.argsort([strip.index for strip in self.strips])
                for strip, new_index in zip(self.strips, new_indices):
                    strip.index = int(new_index) # avoid json serialization issues

            self.left_extremities = [self(0)]
            self.right_extremities = [self(-1)]
            self.sizes = [len(self.strips)]


    def __call__(self, i):
        ''' Returns the i-th strip. '''

        return self.strips[i]


    def __add__(self, other):
        ''' Including new strips. '''

        N = len(self.strips)
        union = self.copy()
        other = other.copy()
        for strip in other.strips:
            strip.index += N

        union.left_extremities += other.left_extremities
        union.right_extremities += other.right_extremities
        union.strips += other.strips
        return union


    def copy(self):
        ''' Copy object. '''

        return copy.deepcopy(self)


    def size(self):
        ''' Number of strips. '''

        return len(self.strips)


    def sizes(self):
        ''' Number of strips of each added document. '''

        return [(r.index - l.index + 1)for r, l in zip(self.right_extremities, self.left_extremities)]


    def shuffle(self):

        random.shuffle(self.strips)
        return self


    def permutation(self):
        ''' Return the permutation (order) of the strips. '''

        return [strip.index for strip in self.strips]


    def extremities(self):
        ''' Return the ground-truth indices of the strips belonging to the documents' extremities. '''

        left_indices = [strip.index for strip in self.left_extremities]
        right_indices = [strip.index for strip in self.right_extremities]
        return left_indices, right_indices


    def _load_data(self, path, regex_str='.*\d\d\d\d\d\.*'):
        ''' Stack strips horizontally.

        Strips are images with same basename (and extension) placed in a common
        directory. Example:

        basename="D001" and extension=".jpg" => strips D00101.jpg, ..., D00130.jpg.
        '''

        path_images = '{}/strips'.format(path)
        path_masks = '{}/masks'.format(path)
        regex = re.compile(regex_str)

        # loading images
        fnames = sorted([fname for fname in os.listdir(path_images) if regex.match(fname)])
        images = []
        for fname in fnames:
            image = cv2.cvtColor(
                cv2.imread('{}/{}'.format(path_images, fname)),
                cv2.COLOR_BGR2RGB
            )
            images.append(image)

        # load masks
        masks = []
        if os.path.exists(path_masks):
            for fname in fnames:
                mask = np.load('{}/{}.npy'.format(path_masks, os.path.splitext(fname)[0]))
                masks.append(mask)
        else:
            masks = len(images) * [None]
            # self.artificial_mask = True

        for index, (image, mask) in enumerate(zip(images, masks), 1):
            strip = Strip(image, index, mask)
            self.strips.append(strip)


    def image(self, order=None, ground_truth_order=False, displacements=None, filled=False):
        ''' Return the document as an image.

        order: list with indices of the strips (if not None, ground_truth order is ignored).
        ground_truth_order: if True (and order is None), it composes the image in the ground-truth order.
        displacements: relative displacements between neighbors strips.
        filled: if True, the background is white.
        '''

        corrected = []
        if order is None:
            if ground_truth_order:
                corrected = sorted(self.strips, key=lambda x: x.index)
            else:
                corrected = self.strips
        else:
            corrected = [self(idx) for idx in order]

        if displacements is None:
            displacements = len(self.strips) * [0]
        stacked = corrected[0].copy()
        for current, disp in zip(corrected[1 :], displacements):
            stacked.stack(current, disp=disp, filled=filled)
        return stacked.image


    def post_processed_image(self, order=None, ground_truth_order=False, displacements=None, filled=False, delta_y=50):

        # extracting crops
        h = self.image().shape[0]
        crops = []
        for y in range(0, h - delta_y, delta_y):
            crop = self.copy().crop_vertically(y, y + delta_y - 1).image(
                order=order, ground_truth_order=ground_truth_order,
                displacements=displacements, filled=filled
            )
            crops.append(crop)

        # result image
        h = sum([crop.shape[0] for crop in crops])
        w = max([crop.shape[1] for crop in crops])
        result = np.empty((h, w, 3), dtype=np.uint8)
        result[:] = 255 if filled else 0

        # joining crops
        y = 0
        for crop in crops:
            w = crop.shape[1]
            result[y : y + delta_y, : w] = crop
            y += delta_y
        return result


    def crop_vertically(self, y1, y2):
        ''' Crop the strips vertically from y1 to y2. '''
        i = 0
        for strip in self.strips:
            i += 1
            strip.crop_vertically(y1, y2)
        return self