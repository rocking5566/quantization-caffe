import os
import cv2

coco_image_path = '/data/dataset_zoo/coco2017/val2017'


def get_image_id_in_path(img_name):
    ori_name = os.path.splitext(img_name)[0]
    image_id = int(ori_name[ori_name.rfind('_') + 1:])
    return image_id


def coco_generator(generate_count=float('inf'), preprocess=None):
    imgs_name = os.listdir(coco_image_path)
    count = 0

    for img_name in imgs_name:
        if count >= generate_count:
            break
        else:
            count += 1

        img_path = os.path.join(coco_image_path, img_name)
        x = cv2.imread(img_path)

        if preprocess is not None:
            x = preprocess(x)

        image_id = get_image_id_in_path(img_name)
        yield [x, image_id]


if __name__ == '__main__':
    imgs_name = os.listdir(coco_image_path)
    with open("coco_imgs.txt", "w") as out_fp:
        for img_name in imgs_name:
            img_path = os.path.join(coco_image_path, img_name)
            out_fp.write(img_path + '\n')
