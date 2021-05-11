from __future__ import division
import numpy as np


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