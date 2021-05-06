from .gen_imagenet_list import imagenet_generator
import cv2


def file_generator(image_list, generate_count=float('inf'), preprocess=None):
    count = 0
    with open(image_list, 'r') as f_img_list:
        for img_path in f_img_list.readlines():
            if count >= generate_count:
                break
            else:
                count += 1

            img_path = img_path.split('\n')[0]
            x = cv2.imread(img_path)

            if preprocess is not None:
                x = preprocess(x)

            yield [x]
