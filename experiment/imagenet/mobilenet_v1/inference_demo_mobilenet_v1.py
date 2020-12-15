
from __future__ import division
import caffe
import cv2
import numpy as np
import os
from mobilenet_v1_util import preprocess, postprocess

model_path = '/data/models_zoo/imagenet/mobilenet_v1/caffe/2020.04.08.01'
g_caffe_proto = os.path.join(model_path, 'mobilenet_0.25_bnmerge.prototxt')
g_caffe_weight = os.path.join(model_path, 'mobilenet_0.25_bnmerge.caffemodel')
g_caffe_weight = os.path.join(model_path, 'mobilenet_0.25_bnoutlier_bnmerge.caffemodel')

img_path = '/workspace/experiment/imagenet/testpics/husky.jpg'


def inference_from_jpg():
    caffe.set_mode_gpu()
    net = caffe.Net(g_caffe_proto, g_caffe_weight, caffe.TEST)

    img_bgr = cv2.imread(img_path)
    x = preprocess(img_bgr)

    net.blobs['data'].data[...] = x
    out = net.forward()
    y = out['logits']

    for pred in postprocess(y):
        print(pred)

if __name__ == '__main__':
    inference_from_jpg()
