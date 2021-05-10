from __future__ import division
import caffe
import os
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from model_path import get_caffe_model_path


def conv_mac_count(target_layer_list, proto, verbose=True):
    net = caffe.Net(proto, caffe.TEST)
    proto_param = caffe_pb2.NetParameter()
    text_format.Merge(open(proto).read(), proto_param)
    total_mac = 0
    float_mac = 0

    for layer in proto_param.layer:
        if layer.type == 'Convolution':
            x = net.blobs[layer.bottom[0]].data[...]
            y = net.blobs[layer.top[0]].data[...]

            if layer.convolution_param.kernel_size:
                kh = kw = int(layer.convolution_param.kernel_size[0])
            else:
                kw = layer.convolution_param.kernel_w
                kh = layer.convolution_param.kernel_h
            cin = x.shape[1]
            cout = y.shape[1]
            oh = y.shape[2]
            ow = y.shape[3]

            if layer.convolution_param.group != 1:
                mac = kh * kw * oh * ow * cin
            else:
                mac = kh * kw * cin * cout * oh * ow

            total_mac += mac

            if layer.name in target_layer_list:
                float_mac += mac

    if verbose:
        print('float_mac / total_mac = ', float_mac / total_mac)
        print('float_mac = ', float_mac)
        print('total_mac = ', total_mac)

    return float_mac, total_mac


if __name__ == '__main__':
    proto, weight, _ = get_caffe_model_path('yolo_v3')
    net = caffe.Net(proto, weight, caffe.TEST)
    target_layers = ['layer82-conv', 'layer94-conv', 'layer106-conv']
    conv_mac_count(target_layers, proto)
