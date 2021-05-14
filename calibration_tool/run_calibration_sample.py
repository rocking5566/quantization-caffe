import os
import cv2
from model_path import get_caffe_model_path
from experiment.imagenet.mobilenet_v1.mobilenet_v1_util import preprocess
import numpy as np
from kld_calibrator import KLD_Calibrator

g_image_list = '/workspace/util/dataset/imagenet_imgs.txt'
model_name = 'mobilenet_v1_0.25'


if __name__ == '__main__':
    proto, weight, _ = get_caffe_model_path(model_name)

    calibrator = KLD_Calibrator(g_image_list, proto, weight,
        image_count=100, preprocess=preprocess, histogram_bin_num=20480)

    threshold_table = calibrator.do_calibration()
    threshold_table = calibrator.threshold_opt(threshold_table)
    calibrator.dump_threshold_table(threshold_table, '{}.threshold_table'.format(model_name))
