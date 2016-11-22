from pdb import set_trace as debugger
from collections import defaultdict
from PIL import Image

import itertools as it
import numpy as np

import os
import sys
import random

class ImageLoader(object):

    def __init__(self, textfile='./list.txt'):
        self.textfile = textfile

        self.images = None
        self.subjects = None

        self._train_imgs = None
        self._test_imgs = None

        self._xsize = 50

        if not os.path.exists(self.textfile):
            sys.exit('Cannot find list file')
        self.create_split()

    def create_split(self):
        data = open(self.textfile, 'r').readlines()
        self.images = defaultdict(list)
        self.subjects = {}
        subject_id = 0
        for fn in data:
            name = fn.split('/')[3]
            # format is {'dir_name': ['dir/image1', 'dir/image2', ...]}A
            self.images[name].append(fn.strip())
            # save subject ids for later
            if name not in self.subjects:
                self.subjects[name] = subject_id
                subject_id += 1
        # split into training and test sets
        self.train_test_split(self.images)

    def train_test_split(self, _images, split=0.7):
        # flatten, shuffle and split
        total = sum(map(len, _images.values()))
        _imgs_flat = sum(_images.values(), [])
        random.shuffle(_imgs_flat)
        # number of samples for testing and train
        n_train = int(total * split)
        self._train_imgs = _imgs_flat[0:n_train]
        self._test_imgs = _imgs_flat[n_train:]

    def create_pairs(self, batch_size, train=True):
        assert batch_size % 2 == 0, 'Batch size should be an even number'
        data_x1 = np.zeros((batch_size, self._xsize, self._xsize))
        data_x2 = np.zeros((batch_size, self._xsize, self._xsize))
        labels_x1 = np.zeros((batch_size, 1))
        labels_x2 = np.zeros((batch_size, 1))
        # range of indices to pick from
        allowed_range = range(0, batch_size)
        for i in xrange(batch_size/2):
            data_x1, data_x2, labels_x1, labels_x2, allowed_range = self._pick_faces(data_x1, 
                            data_x2, labels_x1, labels_x2, allowed_range, train, same=True)
            data_x1, data_x2, labels_x1, labels_x2, allowed_range = self._pick_faces(data_x1, 
                            data_x2, labels_x1, labels_x2, allowed_range, train, same=False)
        return data_x1, data_x2, labels_x1, labels_x2

    def _pick_faces(self, data_x1, data_x2, labels_x1, 
                   labels_x2, allowed_range, train, same=None):
        indx = random.choice(allowed_range)
        allowed_range.remove(indx)
        # Pick 2 of the same or different faces
        _img1, _img2 = self._pick_same_faces(train) if same else self._pick_diff_faces(train)
        data_x1[indx,:,:] = np.array(self._load_img(_img1))
        labels_x1[indx] = self.subjects[_img1.split('/')[3]]
        data_x2[indx,:,:] = np.array(self._load_img(_img2))
        labels_x2[indx] = self.subjects[_img2.split('/')[3]]
        return data_x1, data_x2, labels_x1, labels_x2, allowed_range

    def _load_img(self, img=None):
        return Image.open(img).resize((self._xsize, self._xsize), Image.ANTIALIAS)

    def _pick_diff_faces(self, train):
        img1 = random.choice(self._train_imgs if train else self._test_imgs)
        name = img1.split('/')[3]
        img2 = None
        while True:
            # pick two images
            rand_name = random.choice(self.images.keys())
            while rand_name == name:
                rand_name = random.choice(self.images.keys())
            img2 = random.choice(self.images[rand_name])
            # make sure they come from right sets
            if img2 not in [self._train_imgs, self._test_imgs][not train]:
                rand_name = random.choice(self.images.keys())
                img2 = random.choice(self.images[rand_name])
                continue
            break
        return img1, img2

    def _pick_same_faces(self, train):
        img1 = random.choice(self._train_imgs if train else self._test_imgs)
        name = img1.split('/')[3]
        while len(self.images[name]) <= 1:
            img1 = random.choice(self._train_imgs if train else self._test_imgs)
            name = img1.split('/')[3]
        img2 = random.choice(self.images[name])
        return img1, img2

    def next_batch(self, batch_size, train=True):
        return self.create_pairs(batch_size, train)


if __name__ == '__main__':
    it = ImageLoader()
    x1, x2, l1, l2 = it.next_batch(10)
