caffe_root = '../'
IMAGE_FILENAMES = 'train_filenames.npy'
IMAGE_DIR = caffe_root + "working/oxford_pet_dataset/"
MEAN_FILE = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy_feature.prototxt'
PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
FEAT_LAYER = 'fc6wi'

import sys
sys.path.insert(0, caffe_root + 'python')
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
filenames = np.load('train_filenames.npy')
N = len(filenames)
net.blobs['data'].reshape(N,3,227,227)
for i in range(N):
    load_image = IMAGE_DIR + filenames[i]
    net.blobs['data'].data[i] = \
        transformer.preprocess('data', caffe.io.load_image(load_image))
net.forward()
features = net.blobs[FEAT_LAYER].data
np.save('train_features.npy', features)
