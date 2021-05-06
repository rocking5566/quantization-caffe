import caffe
from caffe.proto import caffe_pb2
import copy
import cv2
from google.protobuf import text_format
import math
from math_function import cal_sqnr
import numpy as np
import os


def generic_loss(fp32_preds, int8_dequant_preds):
    ret = 0
    neuron_count = 0

    for op_name in fp32_preds:
        fp32_pred = fp32_preds[op_name]
        int8_dequant_pred = int8_dequant_preds[op_name]
        loss = cal_sqnr(fp32_pred, int8_dequant_pred)
        if not math.isinf(loss):
            ret += -loss * fp32_pred.size
            neuron_count += fp32_pred.size

    if ret == 0 and neuron_count == 0:
        return float('-inf')
    else:
        return ret / neuron_count


class LayerLoss(object):
    def __init__(self, proto, weight, quant_table,
                 input_blob_name, quant_type='int8'):
        self.proto = proto
        self.weight = weight
        self.quant_table = quant_table
        self.input_blob_name = input_blob_name
        self.quant_type = quant_type
        assert(quant_type in ['int8'])

    def predict(self, image_list, val_data_count=100, preprocess=None, loss_func=generic_loss):
        caffe.set_mode_gpu()
        net = caffe.Net(self.proto, self.weight, caffe.TEST)
        net.import_activation_range(self.quant_table)
        loss_list = list()
        pred_fp32 = list()

        print('Collect float32 tensor...')
        with open(image_list, 'r') as f_img_list:
            for img_path in f_img_list.readlines():
                if len(pred_fp32) >= val_data_count:
                    break

                img_path = img_path.split('\n')[0]
                x = cv2.imread(img_path)

                if preprocess is not None:
                    x = preprocess(x)

                net.blobs[self.input_blob_name].reshape(
                    1, 3, x.shape[2], x.shape[3])
                net.blobs[self.input_blob_name].data[...] = x
                y = net.forward()
                pred_fp32.append(copy.deepcopy(y))

        proto_param = caffe_pb2.NetParameter()
        text_format.Merge(open(self.proto).read(), proto_param)
        layer_index = 0
        for layer in proto_param.layer:
            layer_index += 1
            layer_name = str(layer.name)
            layer_type = str(layer.type)
            if not net.is_support_quant_by_layer_name(layer_name):
                continue

            # only quantize one layer, the rest of the layer remain float32
            # Hence, we need to backup the original weight first, then recover the weight after net.forward().
            if net.is_support_quant_weight_by_layer_type(layer_type):
                weight = net.params[layer_name][0].data[...].copy()

            if self.quant_type == 'int8':
                net.init_fakequant_int8(layer_name, True)
            else:
                raise Exception(
                    'Not support quant_type {}'.format(self.quant_type))

            print('Calculate loss of {}...'.format(layer_name))
            loss = 0
            data_index = 0
            with open(image_list, 'r') as f_img_list:
                for img_path in f_img_list.readlines():
                    if data_index >= val_data_count:
                        break

                    img_path = img_path.split('\n')[0]
                    x = cv2.imread(img_path)

                    if preprocess is not None:
                        x = preprocess(x)

                    net.blobs[self.input_blob_name].reshape(
                        1, 3, x.shape[2], x.shape[3])
                    net.blobs[self.input_blob_name].data[...] = x
                    y = net.forward()
                    loss += loss_func(pred_fp32[data_index], y)
                    data_index += 1

            loss_list.append(
                (layer.name, layer_index, layer.type, loss / val_data_count))

            # Reset inference type from fakequant to native
            # Restore weight before fakequant weight
            net.init_all_infer_type_to_native()
            if net.is_support_quant_weight_by_layer_type(layer_type):
                net.params[layer_name][0].data[...] = weight

        loss_list = sorted(loss_list, cmp=lambda x,
                           y: cmp(x[3], y[3]), reverse=True)

        print('{:>12}\t{:>8}\t{:>12}\t{:>20}'.format(
            'Layer Name', 'Layer ID', 'Layer Type', 'Loss'))
        for layer_loss in loss_list:
            print('{:>12}\t{:>8}\t{:>12}\t{:>20}'.format(
                layer_loss[0], layer_loss[1], layer_loss[2], layer_loss[3]))
