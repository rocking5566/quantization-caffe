from __future__ import division
import numpy as np


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
