from __future__ import division
import numpy as np


def fp32_int8_init(net, fp32_layers=[], perchanel=True):
    for layer_name in net._layer_names:
        if layer_name not in fp32_layers:
            net.init_fakequant_int8(layer_name, perchanel)

def fp32_int8_init_from_file(net, fp32_layers_file, perchanel=True):
    fp32_layers = np.loadtxt(fp32_layers_file, str, delimiter='\t')
    fp32_int8_init(net, fp32_layers, perchanel)
