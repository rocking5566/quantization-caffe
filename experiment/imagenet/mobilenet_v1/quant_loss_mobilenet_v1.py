from layer_loss import LayerLoss
from mobilenet_v1_util import preprocess
import os

g_model_path = '/data/models_zoo/imagenet/mobilenet_v1/caffe/2020.04.08.01'
g_caffe_proto = os.path.join(g_model_path, 'mobilenet_0.25_bnmerge.prototxt')
g_caffe_weight = os.path.join(g_model_path, 'mobilenet_0.25_bnmerge.caffemodel')
g_quant_table_path = '/data/models_zoo/imagenet/mobilenet_v1/caffe_int8/2020.05.20.02/bmnet_custom_calibration_table.threshold_table'

g_image_list = '/workspace/util/dataset/imagenet_imgs.txt'
g_val_data_count = 100


if __name__ == '__main__':
    loss_evaluator = LayerLoss(g_caffe_proto, g_caffe_weight, g_quant_table_path, 'data',
                               quant_type='int8')
    loss_evaluator.predict(image_list=g_image_list,
                           val_data_count=g_val_data_count,
                           preprocess=preprocess)
