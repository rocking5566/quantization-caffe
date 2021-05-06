import caffe
import cv2
import os
from model_path import get_caffe_model_path
from retinaface_util import RetinaFace

g_obj_threshold = 0.5
g_nms_threshold = 0.4
g_img_path = '/workspace/testpics/probe.jpg'
retinaface_w, retinaface_h = 600, 600

def inference_from_jpg():
    detector = RetinaFace()
    img = cv2.imread(g_img_path)
    x = detector.preprocess(img, retinaface_w, retinaface_h)

    proto, weight, _ = get_caffe_model_path('retinaface_mnet_0.25')
    net = caffe.Net(proto, weight, caffe.TEST)
    net.blobs['data'].reshape(1, 3, x.shape[2], x.shape[3])
    net.blobs['data'].data[...] = x
    y = net.forward()
    faces, landmarks = detector.postprocess(y, retinaface_w, retinaface_h)
    draw_image = detector.draw(img, faces, landmarks, True)

    cv2.imshow('face', draw_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    caffe.set_mode_gpu()
    inference_from_jpg()


