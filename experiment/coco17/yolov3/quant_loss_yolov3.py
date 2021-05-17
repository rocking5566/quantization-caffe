from layer_loss import LayerLoss
from yolov3_util import preprocess
import math
from math_function import cal_sigmoid, cal_sqnr
from model_path import get_caffe_model_path
import numpy as np
import os

g_image_list = '/workspace/util/dataset/coco_imgs.txt'
g_val_data_count = 100


def yolo_loss(fp32_preds, int8_dequant_preds, yolo_w, yolo_h, yolo_config):
    ret = 0
    effective_loss = 0

    def partial_yolo_postprocess(pred):
        num_boxes_per_cell = 3
        num_of_class = 80

        grid_size = pred.shape[2]
        out = np.transpose(pred, (0, 2, 3, 1))
        out = np.reshape(out, (grid_size, grid_size,
                               num_boxes_per_cell, 5 + num_of_class))
        return out, grid_size

    def yolo_bbox_loc(pred, yolo_w, yolo_h, anchors=None):
        out, grid_size = partial_yolo_postprocess(pred)

        if anchors == None:
            return out[..., 0:4]

        anchors_tensor = np.array(anchors).reshape(1, 1, 3, 2)
        box_xy = cal_sigmoid(out[..., :2])
        box_wh = np.exp(out[..., 2:4]) * anchors_tensor

        col = np.tile(np.arange(0, grid_size),
                      grid_size).reshape(-1, grid_size)
        row = np.tile(np.arange(0, grid_size).reshape(-1, 1), grid_size)

        col = col.reshape(grid_size, grid_size, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_size, grid_size, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_size, grid_size)
        box_wh /= (yolo_w, yolo_h)

        boxes = np.concatenate((box_xy, box_wh), axis=-1)
        return boxes.flatten()

    def bbox_confidence(pred):
        out, _ = partial_yolo_postprocess(pred)
        return out[..., 4].flatten()

    def bbox_class(pred):
        out, _ = partial_yolo_postprocess(pred)
        return out[..., 5:].flatten()

    for op_name, anchors in yolo_config:
        fp32_pred = yolo_bbox_loc(fp32_preds[op_name], yolo_w, yolo_h, anchors)
        int8_dequant_pred = yolo_bbox_loc(
            int8_dequant_preds[op_name], yolo_w, yolo_h, anchors)
        # fp32_pred = bbox_confidence(fp32_preds[op_name])
        # int8_dequant_pred = bbox_confidence(int8_dequant_preds[op_name])
        # fp32_pred = bbox_class(fp32_preds[op_name])
        # int8_dequant_pred = bbox_class(int8_dequant_preds[op_name])
        loss = cal_sqnr(fp32_pred, int8_dequant_pred)
        if not math.isinf(loss):
            ret += -loss
            effective_loss += 1

    return ret / effective_loss


def yolov3_160_loss(fp32, int8_dequant):
    yolo_h, yolo_w = 160, 160
    yolo_config = [('layer82-conv', [116, 90, 156, 198, 373, 326]),
                   ('layer94-conv', [30, 61, 62, 45, 59, 119]),
                   ('layer106-conv', [10, 13, 16, 30, 33, 23])]
    return yolo_loss(fp32, int8_dequant, yolo_w, yolo_h, yolo_config)


def preprocess_160(img_bgr):
    yolo_h, yolo_w = 160, 160
    x = preprocess(img_bgr, yolo_h, yolo_w)
    return np.expand_dims(x, axis=0)


if __name__ == '__main__':
    proto, weight, quant_info = get_caffe_model_path('yolo_v3')
    loss_evaluator = LayerLoss(proto, weight, quant_info, 'data',
                               quant_type='int8')
    loss_evaluator.predict(image_list=g_image_list,
                           val_data_count=g_val_data_count,
                           preprocess=preprocess_160,
                           loss_func=yolov3_160_loss)
