
from __future__ import division
import caffe
import cv2
import numpy as np
import os
from mobilenet_v1_util import preprocess, postprocess
from model_path import get_caffe_model_path

img_path = '/workspace/testpics/husky.jpg'


def inference_from_jpg():
    caffe.set_mode_gpu()
    proto, weight, quant_info = get_caffe_model_path('mobilenet_v1_0.25')
    net = caffe.Net(proto, weight, caffe.TEST)
    # net.import_activation_range(quant_info)
    # net.init_all_fakequant_int8(True)
    # fp32_int8_init_from_file(net, 'float_layers.txt')
    net.PrintQuantInfo()

    img_bgr = cv2.imread(img_path)
    x = preprocess(img_bgr)

    net.blobs['data'].data[...] = x
    out = net.forward()
    y = out['logits']

    for pred in postprocess(y):
        print(pred)


if __name__ == '__main__':
    inference_from_jpg()
