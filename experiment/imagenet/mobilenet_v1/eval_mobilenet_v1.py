from __future__ import division

import caffe
from caffe.proto import caffe_pb2
import cv2
from google.protobuf import text_format
import lmdb
import numpy as np
import os
from mobilenet_v1_util import preprocess, postprocess
from model_path import get_caffe_model_path

data_path = '/data/dataset_zoo/imagenet/ilsvrc12_256'
nh, nw = 224, 224
batch_size = 50


def print_result(image_count, top1_correct_count, top5_correct_count):
    print('image_count', image_count)
    print('top1_correct_count', top1_correct_count)
    print('top5_correct_count', top5_correct_count)
    print('top1 precision', top1_correct_count / image_count)
    print('top5 precision', top5_correct_count / image_count)

if __name__ == "__main__":
    caffe.set_mode_gpu()
    proto, weight, quant_info = get_caffe_model_path('mobilenet_v1_0.25')
    net = caffe.Net(proto, weight, caffe.TEST)
    net.PrintQuantInfo()

    net.blobs['data'].reshape(batch_size, 3, nh, nw)

    image_count = 0
    top1_correct_count = 0
    top5_correct_count = 0
    gt_labels = np.zeros(batch_size, dtype=int)

    for i in range(1001):
        class_path = os.path.join(data_path, str(i))
        imgs_name = os.listdir(class_path)

        for img_name in imgs_name:
            image_path = os.path.join(class_path, img_name)
            img_bgr = cv2.imread(image_path)
            x = preprocess(img_bgr)
            net.blobs['data'].data[image_count % batch_size][...] = x
            gt_labels[image_count % batch_size] = i
            image_count += 1

            # FIXME - only inference complete batch so far
            if image_count % batch_size == 0:
                out = net.forward()
                y = out['logits']
                y = np.squeeze(y)

                if batch_size == 1:
                    y = y.reshape((1, 1000))

                # CAUSION - data_count % batch_size must == 0
                for j in range(batch_size):
                    top_5 = np.argsort(-y[j])[:5]
                    if top_5[0] == gt_labels[j]:
                        top1_correct_count += 1

                    if np.any(top_5[0:5] == gt_labels[j]):
                        top5_correct_count += 1

                print_result(image_count, top1_correct_count, top5_correct_count)

    if image_count > 0:
        print_result(image_count, top1_correct_count, top5_correct_count)
