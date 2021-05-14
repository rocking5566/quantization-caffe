import caffe
import cv2
from dataset.gen_coco_list import coco_generator
import json
from model_path import get_caffe_model_path
import numpy as np
import os
from yolov3_util import preprocess, postprocess, draw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_annotation = '/data/dataset_zoo/coco2017/annotations/instances_val2017.json'
coco_result_file = './coco_results.json'

g_obj_threshold = 0.005
g_nms_threshold = 0.45
g_num_of_class = 80
g_yolo_h, g_yolo_w = 160, 160


def clip_box(box, image_shape):
    x, y, w, h = box
    xmin = max(0, x)
    ymin = max(0, y)
    xmax = min(image_shape[1], x + w)
    ymax = min(image_shape[0], y + h)

    bx = xmin
    by = ymin
    bw = xmax - xmin
    bh = ymax - ymin
    return np.array([bx, by, bw, bh])

def inference_yolo_on_coco():
    coco_ids= [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17,
               18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
               37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
               54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
               74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    caffe.set_mode_gpu()
    proto, weight, _ = get_caffe_model_path('yolo_v3')
    net = caffe.Net(proto, weight, caffe.TEST)
    net.blobs['data'].reshape(1, 3, g_yolo_h, g_yolo_w)

    jdict = []
    img_ids = []
    data_count = 0
    for [img_bgr, image_id] in coco_generator():
        img_ids.append(image_id)
        x = preprocess(img_bgr, g_yolo_h, g_yolo_w)
        net.blobs['data'].data[...] = x
        y = net.forward()
        batched_pred = postprocess(y, img_bgr.shape, g_yolo_h, g_yolo_w, g_num_of_class, g_obj_threshold, g_nms_threshold)

        for pred in batched_pred[0]:
            x, y, w, h = clip_box(pred[0], img_bgr.shape)
            score = pred[1]
            cls = pred[2]
            jdict.append({
                "image_id": image_id,
                "category_id": coco_ids[cls],
                "bbox": [x, y, w, h],
                "score": float(score)
            })

        data_count += 1
        print('{} / image_id = {}'.format(data_count, image_id))


    with open(coco_result_file, 'w') as coco_result:
        json.dump(jdict, coco_result)

    coco_gt = COCO(coco_annotation)
    coco_dt = coco_gt.loadRes(coco_result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()



if __name__ == '__main__':
    inference_yolo_on_coco()

