from __future__ import division
import numpy as np


def fp32_int8_init_perchannel(net, fp32_layers=[]):
    for layer_name in net._layer_names:
        if layer_name not in fp32_layers:
            net.init_fakequant_int8(layer_name, True)

def fp32_int8_init_perchannel_from_file(net, fp32_layers_file):
    fp32_layers = np.loadtxt(fp32_layers_file, str, delimiter='\t')
    for layer_name in net._layer_names:
        if layer_name not in fp32_layers:
            net.init_fakequant_int8(layer_name, True)
