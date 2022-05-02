import pickle
import numpy as np

import os
import pdb
import time
import math
import shutil
import scipy.io as sio


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='latin1')
    if 'data' in dict:
        dict['data'] = dict['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2) 
    return dict




def load_cifar10_one(filename):
    """
    load single batch of cifar-10 dataset
    
    code is adapted from CS231n assignment kit
    
    @param filename: string of file name in cifar
    @return: X, Y: data and labels of images in the cifar batch
    """
    
    
    data = unpickle(filename);
        
    X=data['data']
    Y=data['labels']
        

    return X, Y; 


def load_cifar10_set(file):
    data, labels = load_cifar10_one(file+ '1')
    for f in range(1,6):
        data_n, labels_n = load_cifar10_one(file+ str(f))
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    return data, labels



def load_svhn_train(file_name):
    
    data = sio.loadmat(file_name)
    svhn_trn_x = data['X']
    svhn_trn_y = data['y'] - 1
    svhn_trn_x = svhn_trn_x.swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1) 
    svhn_trn_y = np.squeeze(svhn_trn_y)
    
    return svhn_trn_x,svhn_trn_y

def load_svhn_test(file_name):
    data = sio.loadmat(file_name)
    svhn_tst_x = data['X']
    svhn_tst_y = data['y'] - 1
    svhn_tst_x = svhn_tst_x.swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1) 
    svhn_tst_y = np.squeeze(svhn_tst_y)
    
    return svhn_tst_x,svhn_tst_y




def load_cifar10_test(path, inp_size):
    fid = open(path + "images.txt", "r")
    img_names = fid.read().splitlines()
    fid.close()
    tst_img = np.zeros([len(img_names), inp_size, inp_size, 3])
    for m in xrange(len(img_names)):
        data = imread(path + "images/" + img_names[m])
        if len(data.shape) == 2:
            data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        tst_img[m, :, :, :] = imresize(data, (inp_size, inp_size, 3))

    return tst_img