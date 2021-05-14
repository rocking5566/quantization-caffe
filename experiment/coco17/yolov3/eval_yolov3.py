import caffe
from dataset.eval_coco import coco_evaluation
from model_path import get_caffe_model_path
from yolov3_util import preprocess, postprocess, draw

coco_annotation = '/data/dataset_zoo/coco2017/annotations/instances_val2017.json'
coco_result_file = './coco_results.json'
g_obj_threshold = 0.005
g_nms_threshold = 0.45
g_num_of_class = 80
g_yolo_h, g_yolo_w = 160, 160


if __name__ == '__main__':
    caffe.set_mode_gpu()
    proto, weight, quant_info = get_caffe_model_path('yolo_v3')
    net = caffe.Net(proto, weight, caffe.TEST)
    net.blobs['data'].reshape(1, 3, g_yolo_h, g_yolo_w)

    def caffe_detect(img_bgr):
        x = preprocess(img_bgr, g_yolo_h, g_yolo_w)
        net.blobs['data'].data[...] = x
        y = net.forward()
        batched_pred = postprocess(
            y, img_bgr.shape, g_yolo_h, g_yolo_w, g_num_of_class, g_obj_threshold, g_nms_threshold)

        return batched_pred[0]

    coco_evaluation(caffe_detect)