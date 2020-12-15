
from __future__ import division
import cv2
import numpy as np

rgb_mean = [0.485, 0.456, 0.406] # rgb order
rgb_std = [0.229, 0.224, 0.225] # rgb order
nh, nw = 224, 224


def crop_center(img, crop_w, crop_h):
    h, w, _ = img.shape
    offset_h = int((h - crop_h) / 2)
    offset_w = int((w - crop_w) / 2)
    return img[offset_h:h - offset_h, offset_w:w - offset_w]

def preprocess(img_bgr):
    img_bgr = crop_center(img_bgr, nw, nh)
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = np.transpose(img_bgr, (2, 0, 1))  # HWC to CHW
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32, copy=False)
    x /= 255.
    x[:, 0, :, :] -= rgb_mean[0]
    x[:, 1, :, :] -= rgb_mean[1]
    x[:, 2, :, :] -= rgb_mean[2]
    x[:, 0, :, :] /= rgb_std[0]
    x[:, 1, :, :] /= rgb_std[1]
    x[:, 2, :, :] /= rgb_std[2]
    return x

def postprocess(y, top=5):
    imagenet_label_path = '/workspace/experiment/imagenet/testpics/imagenet_synset_to_human_label_map.txt'
    pred = np.squeeze(y)
    e_pred = np.exp(pred - np.max(pred))
    softmax_pred = e_pred / e_pred.sum()
    idx = np.argsort(-softmax_pred)

    label_names = np.loadtxt(imagenet_label_path, str, delimiter='\t')
    label_name_prob = []

    for i in range(top):
      label = idx[i]
      label_name_prob.append((label_names[label], softmax_pred[label]))

    return label_name_prob