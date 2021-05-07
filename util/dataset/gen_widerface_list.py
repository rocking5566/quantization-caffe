import argparse
import cv2
import os
import scipy.io


def widerface_generator(wider_face_label, wider_face_image, preprocess=None):
    f = scipy.io.loadmat(wider_face_label)
    event_list = f.get('event_list')
    file_list = f.get('file_list')
    face_bbx_list = f.get('face_bbx_list')

    for event_idx, event in enumerate(event_list):
        directory = event[0][0]
        for im_idx, im in enumerate(file_list[event_idx][0]):
            im_name = im[0][0]
            face_bbx = face_bbx_list[event_idx][0][im_idx][0]
            bboxes = []

            for i in range(face_bbx.shape[0]):
                xmin = int(face_bbx[i][0])
                ymin = int(face_bbx[i][1])
                xmax = int(face_bbx[i][2]) + xmin
                ymax = int(face_bbx[i][3]) + ymin
                bboxes.append((xmin, ymin, xmax, ymax))

            image_path = os.path.join(wider_face_image, directory, im_name + '.jpg')
            img = cv2.imread(image_path)
            if preprocess is not None:
                img = preprocess(img)

            yield [img, bboxes, directory, im_name]


if __name__ == '__main__':
    g_wider_face_path = '/data/dataset_zoo/widerface/'
    g_img_path = os.path.join(g_wider_face_path, 'WIDER_val/images')
    g_wider_face_label = \
        os.path.join(g_wider_face_path, 'wider_face_split/wider_face_val.mat')

    with open("widerface_imgs.txt", "w") as out_fp:
        for image, bboxes, img_type, img_name in widerface_generator(g_wider_face_label, g_img_path):
            image_path = os.path.join(g_img_path, img_type, img_name + '.jpg')
            out_fp.write(image_path + '\n')
            # for bbox in bboxes:
            #     cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            # cv2.imshow('widerface', image)
            # print(img_type, img_name)
            # ch = cv2.waitKey(0) & 0xFF
            # if ch == 27:
            #     break
