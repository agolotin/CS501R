#! /usr/bin/env python

from pdb import set_trace as debugger

import sys
import numpy as np
import pandas

def upload_cifar(filename):
    import cPickle
    fo = open(filename, 'rb')
    _dict = cPickle.load(fo)
    fo.close()

    features = data['data']
    labels = data['labels']
    labels = np.atleast_2d( labels ).T
     
    # squash classes 0-4 into class 0, and squash classes 5-9 into class 1
    labels[ labels < 5 ] = 0
    labels[ labels >= 5 ] = 1
    return features, labels

def upload_iris(filename):
    data = pandas.read_csv(filename)
    debugger()
    m = data.as_matrix()
    labels = m[:,0]
    labels[ labels==2 ] = 1  # squash class 2 into class 1
    labels = np.atleast_2d( labels ).T
    features = m[:,1:5]
    return features, labels



if __name__ == '__main__':
    if 'cifar' in sys.argv[1].lower():
        features, labels = upload_cifar(sys.argv[1])
    else:
        features, labels = upload_iris(sys.argv[1])
