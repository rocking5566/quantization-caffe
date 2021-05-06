import caffe
import cv2
import os
from model_path import get_caffe_model_path
from yolov3_util import preprocess, postprocess, draw

img_path = '/workspace/experiment/coco17/testpics/fish-bike.jpg'
labelmap_file = '/workspace/experiment/coco17/labelmap_coco.prototxt'

g_obj_threshold = 0.3
g_nms_threshold = 0.5
g_num_of_class = 80
g_yolo_h, g_yolo_w = 320, 320


def inference_from_jpg():
    proto, weight, _ = get_caffe_model_path('yolo_v3')
    net = caffe.Net(proto, weight, caffe.TEST)
    net.blobs['data'].reshape(1, 3, g_yolo_h, g_yolo_w)
    caffe.set_mode_gpu()

    img = cv2.imread(img_path)
    x = preprocess(img, g_yolo_h, g_yolo_w)

    net.blobs['data'].data[...] = x
    y = net.forward()

    pred = postprocess(y, img.shape, g_yolo_h, g_yolo_w, g_num_of_class, g_obj_threshold, g_nms_threshold)
    for i in range(len(pred)):
        if len(pred[i]) != 0:
            img = draw(img, pred[i], labelmap_file, verbose=True)
            cv2.imshow('yolov3 detect', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    inference_from_jpg()
