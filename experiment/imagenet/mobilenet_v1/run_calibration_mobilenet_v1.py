import os
import cv2
from model_path import get_caffe_model_path
from mobilenet_v1_util import preprocess
import numpy as np
from base_calibrator import Base_Calibrator

g_image_list = '/workspace/util/dataset/imagenet_imgs.txt'


if __name__ == '__main__':
    proto, weight, _ = get_caffe_model_path('mobilenet_v1_0.25')
    calibrator = Base_Calibrator(g_image_list, proto, weight,
        image_count=100, preprocess=preprocess)
    tensor_min_max_dict = calibrator.do_calibration()

    for layer_name, thr in tensor_min_max_dict.items():
        print(layer_name, thr)
