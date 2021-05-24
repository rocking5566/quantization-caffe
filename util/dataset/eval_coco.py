from dataset.gen_coco_list import coco_generator
import json
import numpy as np
import random
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_annotation = '/data/dataset_zoo/coco2017/annotations/instances_val2017.json'
coco_result_file = './coco_results{}.json'.format(random.randint(0, 1000))
g_val_data_count = 5000


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


def coco_evaluation(detect_func):
    coco_ids = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
                54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
                74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    jdict = []
    img_ids = []
    pbar = tqdm(coco_generator(generate_count=g_val_data_count),
                total=g_val_data_count)

    for [img_bgr, image_id] in pbar:
        img_ids.append(image_id)
        bboxs = detect_func(img_bgr)

        for bbox in bboxs:
            x, y, w, h = clip_box(bbox[0], img_bgr.shape)
            score = bbox[1]
            cls = bbox[2]
            jdict.append({
                "image_id": image_id,
                "category_id": coco_ids[cls],
                "bbox": [x, y, w, h],
                "score": float(score)
            })

    with open(coco_result_file, 'w') as coco_result:
        json.dump(jdict, coco_result)

    coco_gt = COCO(coco_annotation)
    coco_dt = coco_gt.loadRes(coco_result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
