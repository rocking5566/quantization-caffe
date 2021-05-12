import cv2
import numpy as np
import os
import math
from base_calibrator import Base_Calibrator
from ctypes import *
from tqdm import tqdm


class KLD_Calibrator(Base_Calibrator):
    def __init__(self,
                 image_list,
                 proto,
                 weight,
                 preprocess=None,
                 image_count=None,
                 input_blob_name='data',
                 is_symmetric_quantization=True,
                 histogram_bin_num=2048,
                 math_lib_path=os.path.join(os.environ['BUILD_ROOT'], 'calibration_tool', 'calibration_math.so')):
        super(KLD_Calibrator, self).__init__(image_list, proto, weight,
                                             preprocess, image_count, input_blob_name, is_symmetric_quantization)
        if not self.is_symmetric_quantization:
            raise RuntimeError(
                "KLD_Calibrator only support symmetric quantization")
        self.histogram_bin_num = int(histogram_bin_num)
        self.calibration_math = CDLL(math_lib_path)
        self.calibration_math.kl_diversity.restype = c_float
        self.calibration_math.kl_diversity_hist.restype = c_float

    def KLD_hist(self, data, width):
        return self.calibration_math.kl_diversity_hist(
            data.ctypes.data_as(POINTER(c_int)), c_float(width),
            c_longlong(self.histogram_bin_num))

    def do_histogram(self, data_max):
        data_hist = {}
        width_hist = {}
        pbar = tqdm(self.images, total=self.image_count,
                    position=0, leave=True)
        for img_id, img_path in enumerate(pbar):
            pbar.set_description(
                "calculate histogram: {}".format(img_path.split("/")[-1]))
            pbar.update(1)
            if img_id >= self.image_count:
                break

            x = cv2.imread(img_path)
            if self.preprocess is not None:
                x = self.preprocess(x)

            self.net.blobs[self.input_blob_name].data[...] = x
            self.net.forward()

            for layer_id, layer_name in enumerate(self.net._layer_names):
                if self.net.is_support_quant_by_layer_name(layer_name):
                    top_blob_id = self.net._top_ids(layer_id)[0]
                    top_name = self.net._blob_names[top_blob_id]
                    activation = self.net.blobs[top_name].data
                    t = np.abs(activation.flatten())
                    t = t[t != 0]

                    width = data_max[layer_name] / (self.histogram_bin_num - 1)
                    if t.size > 0:
                        hist, _ = np.histogram(np.floor(t / width + 0.5),
                                               bins=self.histogram_bin_num,
                                               range=(
                            0, self.histogram_bin_num-1),
                            density=False)
                    else:
                        hist = np.zeros(self.histogram_bin_num)

                    hist = hist.astype(np.int32)
                    if layer_name not in data_hist:
                        data_hist[layer_name] = hist
                        width_hist[layer_name] = width
                    else:
                        data_hist[layer_name] += hist

        pbar.close()
        return data_hist, width_hist

    def do_calibration(self):
        base_thresholds = super(KLD_Calibrator, self).do_calibration()
        data_hist, width_hist = self.do_histogram(base_thresholds)

        thresholds = {}
        pbar = tqdm(data_hist, total=len(data_hist), position=0, leave=True)
        for layer_name in pbar:
            pbar.set_description(
                "calculate threshold from kld histogram: {}".format(layer_name))
            pbar.update(1)
            thresholds[layer_name] = self.KLD_hist(
                data_hist[layer_name], width_hist[layer_name])

        return thresholds
