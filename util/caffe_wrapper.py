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

    def conv_mac_count(self, target_layer_list, verbose=True):
        total_mac = 0
        target_mac = 0

        for layer_id, layer_name in enumerate(self.net._layer_names):
            layer_type = self.layer_type(layer_id)
            if layer_type == 'Convolution':
                # OIHW
                w = self.net.params[layer_name][0].data[...]
                # NCHW
                x = self.net.blobs[self.bottom_blobs_name(layer_id)[
                    0]].data[...]
                y = self.net.blobs[self.top_blobs_name(layer_id)[0]].data[...]

                kh = w.shape[2]
                kw = w.shape[3]
                cin = x.shape[1]
                cout = y.shape[1]
                oh = y.shape[2]
                ow = y.shape[3]

                if cin != w.shape[1]:
                    if (w.shape[1] != 1):
                        raise RuntimeError(
                            "Only support depthwise convolution")
                    # Depthwise Convolution
                    mac = kh * kw * oh * ow * cin
                else:
                    mac = kh * kw * cin * cout * oh * ow

                total_mac += mac

                if layer_name in target_layer_list:
                    target_mac += mac

        if verbose:
            print('target layer = {}'.format(target_layer_list))
            print('target_mac / total_mac = {}'.format(target_mac / total_mac))
            print('target_mac = {}'.format(target_mac))
            print('total_mac = {}'.format(total_mac))

        return target_mac, total_mac

    def int4_8_init(self, quant_info, perchanel=True):
        self.net.import_activation_range(quant_info)
        for layer_name in self.net._layer_names:
            self.net.init_fakequant_int4_8(layer_name, perchanel)
        self.net.PrintQuantInfo()

    def int8_init(self, quant_info, perchanel=True):
        self.net.import_activation_range(quant_info)
        for layer_name in self.net._layer_names:
            self.net.init_fakequant_int8(layer_name, perchanel)
        self.net.PrintQuantInfo()

    def fp32_int8_init(self, quant_info, fp32_layers=[], perchanel=True):
        self.net.import_activation_range(quant_info)
        for layer_name in self.net._layer_names:
            if layer_name not in fp32_layers:
                self.net.init_fakequant_int8(layer_name, perchanel)
        self.net.PrintQuantInfo()
        self.conv_mac_count(fp32_layers)

    def fp32_int8_init_from_file(self, quant_info, fp32_layers_file, perchanel=True):
        self.net.import_activation_range(quant_info)
        fp32_layers = np.loadtxt(fp32_layers_file, str, delimiter='\t')
        self.fp32_int8_init(quant_info, fp32_layers, perchanel)

    def quant_init(self, quant_info, int8_layers=[], int4_8_layer=[], perchanel=True):
        self.net.import_activation_range(quant_info)
        for layer_name in self.net._layer_names:
            if layer_name in int4_8_layer:
                self.net.init_fakequant_int4_8(layer_name, perchanel)
            elif layer_name in int8_layers:
                self.net.init_fakequant_int8(layer_name, perchanel)
            else:
                self.net.set_infer_type_to_native(layer_name)
        self.net.PrintQuantInfo()

    def quant_init_from_file(self, quant_info, int8_layers_file, int4_8_layer_file, perchanel=True):
        self.net.import_activation_range(quant_info)
        int8_layers = np.loadtxt(int8_layers_file, str, delimiter='\t')
        int4_8_layer = np.loadtxt(int4_8_layer_file, str, delimiter='\t')
        self.quant_init(quant_info, int8_layers, int4_8_layer, perchanel)


def init_all_infer_type_to_native(net):
    for layer_name in net._layer_names:
        net.set_infer_type_to_native(layer_name)
