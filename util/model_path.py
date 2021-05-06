from __future__ import division
import os


def get_caffe_model_path(model_name):
    proto = None
    weight = None
    quant_info = None

    if model_name == 'mobilenet_v1_0.25':
        model_path = '/data/models_zoo/imagenet/mobilenet_v1/caffe/2020.04.08.01'
        proto = os.path.join(model_path, 'mobilenet_0.25_bnmerge.prototxt')
        weight = os.path.join(model_path, 'mobilenet_0.25_bnmerge.caffemodel')
        quant_info = '/data/models_zoo/imagenet/mobilenet_v1/caffe_int8/2020.05.20.02/bmnet_custom_calibration_table.threshold_table'
    elif model_name == 'yolo_v3':
        model_path = '/data/models_zoo/object_detection/yolo_v3/caffe/2019.11.15.01'
        proto = os.path.join(model_path, 'yolov3_bnmerge.prototxt')
        weight = os.path.join(model_path, 'bnmerge.caffemodel')
        quant_info = '/data/models_zoo/object_detection/yolo_v3/caffe_int8/2020.09.14.02/bmnet_custom_calibration_table.threshold_table'

    return proto, weight, quant_info