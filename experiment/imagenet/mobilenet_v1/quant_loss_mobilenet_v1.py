from layer_loss import LayerLoss
from mobilenet_v1_util import preprocess
from model_path import get_caffe_model_path
import os

g_image_list = '/workspace/util/dataset/imagenet_imgs.txt'
g_val_data_count = 100


if __name__ == '__main__':
    proto, weight, quant_info = get_caffe_model_path('mobilenet_v1_0.25')
    loss_evaluator = LayerLoss(proto, weight, quant_info, 'data',
                               quant_type='int8')
    loss_evaluator.predict(image_list=g_image_list,
                           val_data_count=g_val_data_count,
                           preprocess=preprocess)
