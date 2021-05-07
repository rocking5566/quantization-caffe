import caffe
from model_path import get_caffe_model_path
import numpy as np
import os
from retinaface_util import RetinaFace
from dataset.eval_widerface import detect_on_widerface, evaluation

g_wider_face_path = '/data/dataset_zoo/widerface/'
g_img_path = os.path.join(g_wider_face_path, 'WIDER_val/images')
g_wider_face_label = \
    os.path.join(g_wider_face_path, 'wider_face_split/wider_face_val.mat')

proto, weight, _ = get_caffe_model_path('retinaface_mnet_0.25')
g_net = caffe.Net(proto, weight, caffe.TEST)
g_nms_threshold = 0.4
g_obj_threshold = 0.005
g_detector = RetinaFace(g_nms_threshold, g_obj_threshold)
g_result_path = './result'


def caffe_detect(img_bgr):
    retinaface_w, retinaface_h = 600, 600
    x = g_detector.preprocess(img_bgr, retinaface_w, retinaface_h)

    g_net.blobs['data'].reshape(1, 3, x.shape[2], x.shape[3])
    g_net.blobs['data'].data[...] = x
    y = g_net.forward()
    faces, _ = g_detector.postprocess(y, retinaface_w, retinaface_h)
    ret = np.zeros(faces.shape)

    for i in range(faces.shape[0]):
        ret[i][0] = faces[i][0]
        ret[i][1] = faces[i][1]
        ret[i][2] = faces[i][2] - faces[i][0]
        ret[i][3] = faces[i][3] - faces[i][1]
        ret[i][4] = faces[i][4]

    return ret


if __name__ == '__main__':
    caffe.set_mode_gpu()
    detect_on_widerface(g_img_path, g_wider_face_label, g_result_path, caffe_detect)
    evaluation(g_result_path, 'retinaface')


