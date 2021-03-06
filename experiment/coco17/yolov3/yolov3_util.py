import cv2
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np


def _correct_boxes(predictions, image_shape, yolo_h, yolo_w):
    image_shape = np.array((image_shape[1], image_shape[0]))
    input_shape = np.array([float(yolo_w), float(yolo_h)])
    new_shape = np.floor(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    correct = []
    for prediction in predictions:
        x, y, w, h = prediction[0:4]
        box_xy = np.array([x, y])
        box_wh = np.array([w, h])
        score = prediction[4]
        cls = int(prediction[5])

        box_xy = (box_xy - offset) * scale
        box_wh = box_wh * scale

        box_xy = box_xy - box_wh / 2.
        box = np.concatenate((box_xy, box_wh), axis=-1)
        box *= np.concatenate((image_shape, image_shape), axis=-1)
        correct.append([box, score, cls])
    return correct


def _softmax(x, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x / np.min(x) * t
    exp_x = np.exp(x)
    out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return out


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _process_feats(out, yolo_h, yolo_w, anchors, num_of_class, obj_threshold):
    grid_size = out.shape[2]
    num_boxes_per_cell = 3

    out = np.transpose(out, (1, 2, 0))
    out = np.reshape(out, (grid_size, grid_size, num_boxes_per_cell, 5 + num_of_class))
    threshold_predictions = []

    anchors_tensor = np.array(anchors).reshape(1, 1, 3, 2)

    box_xy = _sigmoid(out[..., :2])
    box_wh = np.exp(out[..., 2:4]) * anchors_tensor

    box_confidence = _sigmoid(out[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = _softmax(out[..., 5:])

    col = np.tile(np.arange(0, grid_size), grid_size).reshape(-1, grid_size)
    row = np.tile(np.arange(0, grid_size).reshape(-1, 1), grid_size)

    col = col.reshape(grid_size, grid_size, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_size, grid_size, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_size, grid_size)
    box_wh /= (yolo_w, yolo_h)

    boxes = np.concatenate((box_xy, box_wh), axis=-1)

    box_score = box_confidence * box_class_probs
    box_classes = np.argmax(box_score, axis=-1)
    box_class_score = np.max(box_score, axis=-1)

    pos = np.where(box_class_score >= obj_threshold)

    boxes = boxes[pos]
    scores = box_class_score[pos]
    scores = np.expand_dims(scores, axis=-1)
    classes = box_classes[pos]
    classes = np.expand_dims(classes, axis=-1)
    if boxes is not None:
        threshold_predictions = np.concatenate((boxes, scores, classes), axis=-1)

    return threshold_predictions


def _iou(box1, box2):
    inter_left_x = max(box1[0], box2[0])
    inter_left_y = max(box1[1], box2[1])
    inter_right_x = min(box1[0] + box1[2], box2[0] + box2[2])
    inter_right_y = min(box1[1] + box1[3], box2[1] + box2[3])

    if box1[0] == box2[0] and box1[1] == box2[1] and box1[2] == box2[2] and box1[3] == box2[3]:
        return 1.

    inter_w = max(0, inter_right_x - inter_left_x)
    inter_h = max(0, inter_right_y - inter_left_y)

    inter_area = inter_w * inter_h

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    iou = inter_area / (box1_area + box2_area - inter_area)

    return iou


def _non_maximum_suppression(predictions, nms_threshold):
    nms_predictions = list()
    nms_predictions.append(predictions[0])

    i = 1
    while i < len(predictions):
        nms_len = len(nms_predictions)
        keep = True
        j = 0
        while j < nms_len:
            current_iou = _iou(predictions[i][0], nms_predictions[j][0])
            if nms_threshold < current_iou < 1. and predictions[i][2] == nms_predictions[j][2]:
                keep = False

            j = j + 1
        if keep:
            nms_predictions.append(predictions[i])
        i = i + 1

    return nms_predictions


def _postprocess(pred, plot_img_shape, yolo_h, yolo_w, num_of_class, obj_threshold, nms_threshold):
    total_predictions = []
    yolov3_anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    for i, out in enumerate(pred):
        threshold_predictions = _process_feats(out, yolo_h, yolo_w, yolov3_anchors[i], num_of_class, obj_threshold)
        total_predictions.extend(threshold_predictions)

    if not total_predictions:
        return None

    correct_predictions = _correct_boxes(total_predictions, plot_img_shape, yolo_h, yolo_w)
    correct_predictions.sort(key=lambda tup: tup[1], reverse=True)

    nms_predictions = _non_maximum_suppression(correct_predictions, nms_threshold)
    return nms_predictions


def get_label_name(labelmap, labels):
    num_labels = len(labelmap.item)
    label_names = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                label_names.append(labelmap.item[i].display_name.encode('utf-8'))
                break
    return label_names


def draw(image, predictions, labelmap_file, verbose=False):
    image = np.copy(image)
    labelmap = caffe_pb2.LabelMap()
    file = open(labelmap_file, 'r')
    text_format.Merge(str(file.read()), labelmap)
    file.close()

    for prediction in predictions:
        x, y, w, h = prediction[0]
        score = prediction[1]
        cls = prediction[2]

        x1 = max(0, np.floor(x + 0.5).astype(int))
        y1 = max(0, np.floor(y + 0.5).astype(int))

        x2 = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        y2 = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(get_label_name(labelmap, cls), score),
                    (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,255), 1,
                    cv2.LINE_AA)

        if verbose:
            print('class: {0}, score: {1:.2f}'.format(cls, score))
            print('box coordinate x, y, w, h: {0}'.format(prediction[0]))

    return image


def _yolo_v3_output_feature_generator(pred, batch=1):
    layer82_conv = pred['layer82-conv']
    layer94_conv = pred['layer94-conv']
    layer106_conv = pred['layer106-conv']

    for i in range(batch):
        yield [layer82_conv[i], layer94_conv[i], layer106_conv[i]]


def postprocess(pred, plot_img_shape, yolo_h, yolo_w, num_of_class, obj_threshold, nms_threshold, batch=1):
    i = 0
    batch_out = {}

    for _pred in _yolo_v3_output_feature_generator(pred, batch):
        output = _postprocess(_pred, plot_img_shape, yolo_h, yolo_w, num_of_class, obj_threshold, nms_threshold)

        if not output:
            batch_out[i] = []
        else:
            batch_out[i] = output

        i += 1

    return batch_out


def preprocess(bgr_img, yolo_h, yolo_w):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img / 255.0

    ih = rgb_img.shape[0]
    iw = rgb_img.shape[1]

    scale = min(float(yolo_w) / iw, float(yolo_h) / ih)
    rescale_w = int(iw * scale)
    rescale_h = int(ih * scale)

    resized_img = cv2.resize(rgb_img, (rescale_w, rescale_h), interpolation=cv2.INTER_LINEAR)
    new_image = np.full((yolo_h, yolo_w, 3), 0, dtype=np.float32)
    paste_w = (yolo_w - rescale_w) // 2
    paste_h = (yolo_h - rescale_h) // 2

    new_image[paste_h:paste_h + rescale_h, paste_w: paste_w + rescale_w, :] = resized_img
    new_image = np.transpose(new_image, (2, 0, 1))      # row to col, (HWC -> CHW)
    return new_image
