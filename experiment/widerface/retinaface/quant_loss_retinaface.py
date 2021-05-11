from layer_loss import LayerLoss
from retinaface_util import RetinaFace
from model_path import get_caffe_model_path
import os

g_detector = RetinaFace()

g_image_list = '/workspace/util/dataset/widerface_imgs.txt'
g_val_data_count = 100


def preprocess(img_bgr):
    retinaface_w, retinaface_h = 600, 600
    return g_detector.preprocess(img_bgr, retinaface_w, retinaface_h)


if __name__ == '__main__':
    proto, weight, quant_info = get_caffe_model_path('retinaface_mnet_0.25')
    loss_evaluator = LayerLoss(proto, weight, quant_info, 'data',
                               quant_type='int4_8')
    loss_evaluator.predict(image_list=g_image_list,
                           val_data_count=g_val_data_count,
                           preprocess=preprocess)
