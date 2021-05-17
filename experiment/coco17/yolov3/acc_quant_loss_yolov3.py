import caffe
from caffe_wrapper import Net_Helper
from dataset.eval_coco import coco_evaluation
from model_path import get_caffe_model_path
from yolov3_util import preprocess, postprocess, draw

coco_annotation = '/data/dataset_zoo/coco2017/annotations/instances_val2017.json'
coco_result_file = './coco_results.json'
g_obj_threshold = 0.005
g_nms_threshold = 0.45
g_num_of_class = 80
g_yolo_h, g_yolo_w = 160, 160

g_quant_type = 'int4_8'
g_val_data_count = 5000


if __name__ == '__main__':
    caffe.set_mode_gpu()
    proto, weight, quant_info = get_caffe_model_path('yolo_v3')
    net = caffe.Net(proto, weight, caffe.TEST)
    net.import_activation_range(quant_info)
    net_helper = Net_Helper(net)
    net.blobs['data'].reshape(1, 3, g_yolo_h, g_yolo_w)

    def caffe_detect(img_bgr):
        x = preprocess(img_bgr, g_yolo_h, g_yolo_w)
        net.blobs['data'].data[...] = x
        y = net.forward()
        batched_pred = postprocess(
            y, img_bgr.shape, g_yolo_h, g_yolo_w, g_num_of_class, g_obj_threshold, g_nms_threshold)

        return batched_pred[0]

    acc_list = list()
    for layer_id, layer_name in enumerate(net._layer_names):
        layer_type = net_helper.layer_type(layer_id)
        if not net.is_support_quant_by_layer_name(layer_name):
            continue

        # only quantize one layer, the rest of the layer remain float32
        # Hence, we need to backup the original weight first, then recover the weight after net.forward().
        if net.is_support_quant_weight_by_layer_type(layer_type):
            weight = net.params[layer_name][0].data[...].copy()

        if g_quant_type == 'int8':
            net.init_fakequant_int8(layer_name, True)
        elif g_quant_type == 'int4_8':
            net.init_fakequant_int4_8(layer_name, True)
        else:
            raise Exception(
                'Not support quant_type {}'.format(g_quant_type))

        print('Calculate loss of {}...'.format(layer_name))

        result = coco_evaluation(caffe_detect, g_val_data_count)
        map75 = result[2]
        # map50 = result[1]
        acc_list.append((layer_name, layer_id, layer_type, map75))
        print('loss of {} = {}'.format(layer_name, map75))
        # Reset inference type from fakequant to native
        # Restore weight before fakequant weight
        net_helper.init_all_infer_type_to_native()
        if net.is_support_quant_weight_by_layer_type(layer_type):
            net.params[layer_name][0].data[...] = weight

    acc_list = sorted(acc_list, cmp=lambda x,
                        y: cmp(x[3], y[3]), reverse=False)

    print('{:>12}\t{:>8}\t{:>12}\t{:>20}'.format(
            'Layer Name', 'Layer ID', 'Layer Type', 'Loss'))
    for acc in acc_list:
        print('{:>12}\t{:>8}\t{:>12}\t{:>20}'.format(
            acc[0], acc[1], acc[2], acc[3]))