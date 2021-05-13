import abc
import caffe
from caffe_wrapper import Net_Helper
import cv2
import numpy as np
from tqdm import tqdm


class Base_Calibrator(object):
    def __init__(self,
                 image_list,
                 proto,
                 weight,
                 preprocess=None,
                 image_count=None,
                 input_blob_name='data',
                 is_symmetric_quantization=True):
        with open(image_list, 'r') as fp:
            self.images = fp.read().splitlines()

        caffe.set_mode_gpu()
        if len(self.images) == 0:
            raise IOError("ERROR: No calibration image detect.")

        self.image_count = image_count
        self.preprocess = preprocess
        self.net = caffe.Net(proto, weight, caffe.TEST)
        self.input_blob_name = input_blob_name
        self.is_symmetric_quantization = is_symmetric_quantization
        self.tensor_min_max_dict = dict()

        for layer_name in self.net._layer_names:
            self.tensor_min_max_dict[layer_name] = (0, 0)

    def do_calibration(self):
        pbar = tqdm(self.images, total=self.image_count,
                    position=0, leave=True)

        for img_id, img_path in enumerate(pbar):
            pbar.set_description(
                "calculate min and max: {}".format(img_path.split("/")[-1]))
            pbar.update(1)

            if img_id >= self.image_count:
                break

            x = cv2.imread(img_path)
            if self.preprocess is not None:
                x = self.preprocess(x)

            self.net.blobs[self.input_blob_name].data[...] = x
            self.net.forward()

            for layer_id, layer_name in enumerate(self.net._layer_names):
                top_blob_id = self.net._top_ids(layer_id)[0]
                top_name = self.net._blob_names[top_blob_id]
                activation = self.net.blobs[top_name].data

                min_value = np.min(activation)
                max_value = np.max(activation)
                self.tensor_min_max_dict[layer_name] = (
                    min(self.tensor_min_max_dict[layer_name]
                        [0], min_value),
                    max(self.tensor_min_max_dict[layer_name]
                        [1], max_value),
                )

        pbar.close()

        # check max is zero
        for layer_name, (_min, _max) in self.tensor_min_max_dict.items():
            if _max == 0:
                # customer may have network output all zero, change it to 1e-5 for them.
                print("WARNING: layer {} is all zeros. Please check the input data "
                      "correctness.".format(layer_name))
                _max = 1e-5
                self.tensor_min_max_dict[layer_name] = (_min, _max)

            if self.is_symmetric_quantization:
                self.tensor_min_max_dict[layer_name] = max(
                    abs(_min), abs(_max))

        return self.tensor_min_max_dict

    def threshold_opt(self, threshold_table):
        if not self.is_symmetric_quantization:
            raise RuntimeError(
                "threshold_opt only support symmetric quantization")

        threshold_table_opt = dict()
        for k, v in threshold_table.items():
            threshold_table_opt[k] = v

        # TODO - Check pooling.
        # There are global average pooling in mobilenet_v1_0.25.
        # If I remove global average pooling from is_override_prev_threshold
        # Top 1 of mobilenet_v1_0.25 imrpove 0.37694 from 0.37078.
        is_override_prev_threshold = ['Flatten', 'Permute', 'PixelShuffle', 'Reshape', 'Crop', 'ReLU']
        net = Net_Helper(self.net)
        for layer_id, layer_name in enumerate(self.net._layer_names):
            cur_layer_type = net.layer_type(layer_id)
            cur_layer_id = layer_id
            while cur_layer_type in is_override_prev_threshold:
                prev_layer_id = net.get_prev_layer(cur_layer_id)[0]
                prev_layer_name =net.layer_name(prev_layer_id)
                threshold_table_opt[prev_layer_name] = threshold_table_opt[layer_name]
                cur_layer_type = net.layer_type(prev_layer_id)
                cur_layer_id = prev_layer_id

        return threshold_table_opt

    def dump_threshold_table(self, threshold_table, calibration_table_file):
         with open(calibration_table_file, 'w') as writer:
            for layer_name, threshold in threshold_table.items():
                if self.is_symmetric_quantization:
                    line = layer_name + ' ' + str(threshold[0])
                else:
                    line = layer_name + ' ' + str(threshold[0]) + ' ' + str(threshold[1])
                print(line)
                writer.write(line + '\n')
