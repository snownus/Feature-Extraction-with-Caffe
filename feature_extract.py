caffe_root = '../'
image_dir = caffe_root + "working/The Oxford-IIIT Pet Dataset/"
import sys
sys.path.insert(0, caffe_root + 'python')
MEAN_FILE = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy_feature.prototxt'
PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
FEAT_LAYER = 'fc6wi'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', metavar='Inputs', type=str, default='filenames.npy',
                   help='npy filename containing image filenames')
parser.add_argument('-o', metavar='Outputs', type=str, default='features.npy',
                    help='npy filename wirtes extracted features in')
args = parser.parse_args()

import caffe
import numpy as np
caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(MEAN_FILE).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

features = []
IMAGE_FILES = np.load(args.i)
LEN = len(IMAGE_FILES)
net.blobs['data'].reshape(LEN,3,227,227)
for i in range(LEN):
    LOAD_IMAGE = image_dir + IMAGE_FILES[i]
    net.blobs['data'].data[i] = \
        transformer.preprocess('data', caffe.io.load_image(LOAD_IMAGE))
net.forward()
features = net.blobs[FEAT_LAYER].data
np.save(args.o, features)
