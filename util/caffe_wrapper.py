from __future__ import division
import numpy as np


class Net_Helper(object):
    def __init__(self, net):
        self.net = net
        self.blob_source = dict()
        self.layer_name_to_id = dict()

        for layer_id, layer_name in enumerate(self.net._layer_names):
            self.layer_name_to_id[layer_name] = layer_id
            top_blob_ids = self.net._top_ids(layer_id)
            for id in top_blob_ids:
                self.blob_source[id] = layer_id

    def layer_name(self, layer_id):
        return self.net._layer_names[layer_id]

    def layer_id(self, layer_name):
        return self.layer_name_to_id[layer_name]

    def layer_type(self, layer_id):
        return self.net.layers[layer_id].type

    def bottom_blobs_name(self, layer_id):
        bottom_blob_ids = self.net._bottom_ids(layer_id)
        return [self.net._blob_names[id] for id in bottom_blob_ids]

    def top_blobs_name(self, layer_id):
        top_blob_ids = self.net._top_ids(layer_id)
        return [self.net._blob_names[id] for id in top_blob_ids]

    def get_prev_layer(self, layer_id):
        bottom_blob_ids = self.net._bottom_ids(layer_id)
        return [self.blob_source[id] for id in bottom_blob_ids]


def init_all_infer_type_to_native(net):
    for layer_name in net._layer_names:
        net.set_infer_type_to_native(layer_name)


def int4_8_init(net, quant_info, perchanel=True):
    net.import_activation_range(quant_info)
    for layer_name in net._layer_names:
        net.init_fakequant_int4_8(layer_name, perchanel)
    net.PrintQuantInfo()


def int8_init(net, quant_info, perchanel=True):
    net.import_activation_range(quant_info)
    for layer_name in net._layer_names:
        net.init_fakequant_int8(layer_name, perchanel)
    net.PrintQuantInfo()


def fp32_int8_init(net, quant_info, fp32_layers=[], perchanel=True):
    net.import_activation_range(quant_info)
    for layer_name in net._layer_names:
        if layer_name not in fp32_layers:
            net.init_fakequant_int8(layer_name, perchanel)
    net.PrintQuantInfo()


def fp32_int8_init_from_file(net, quant_info, fp32_layers_file, perchanel=True):
    net.import_activation_range(quant_info)
    fp32_layers = np.loadtxt(fp32_layers_file, str, delimiter='\t')
    fp32_int8_init(net, quant_info, fp32_layers, perchanel)


def quant_init(net, quant_info, int8_layers=[], int4_8_layer=[], perchanel=True):
    net.import_activation_range(quant_info)
    for layer_name in net._layer_names:
        if layer_name in int4_8_layer:
            net.init_fakequant_int4_8(layer_name, perchanel)
        elif layer_name in int8_layers:
            net.init_fakequant_int8(layer_name, perchanel)
        else:
            net.set_infer_type_to_native(layer_name)
    net.PrintQuantInfo()


def quant_init_from_file(net, quant_info, int8_layers_file, int4_8_layer_file, perchanel=True):
    net.import_activation_range(quant_info)
    int8_layers = np.loadtxt(int8_layers_file, str, delimiter='\t')
    int4_8_layer = np.loadtxt(int4_8_layer_file, str, delimiter='\t')
    quant_init(net, quant_info, int8_layers, int4_8_layer, perchanel)
