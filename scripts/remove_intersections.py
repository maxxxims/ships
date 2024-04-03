from pathlib import Path
import os
import shutil
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
import json



IMG_SIZE = 1280
MASK_VALUE = 1


def draw_one_box(img, xc, yc, h, w):
    x_left = int(IMG_SIZE * (xc - h / 2))
    y_left = int(IMG_SIZE * (yc - w / 2))
    x_right = int(IMG_SIZE * (xc + h / 2))
    y_right = int(IMG_SIZE * (yc + w / 2))
    #img = cv2.rectangle(img, (x_left, y_left), (x_right, y_right), (1, 0, 0), 1)
    img[y_left:y_right, x_left:x_right] += MASK_VALUE


def calc_iou(img):
    iou = np.sum(img > MASK_VALUE) / np.sum(img == MASK_VALUE)
    return iou


def calc_all_iou(annotation):
    img = np.zeros((IMG_SIZE, IMG_SIZE))
    for el in annotation:
        draw_one_box(img, el[1], el[2], el[3], el[4])
    iou = calc_iou(img)
    return iou

def calc_mean_iou(annotation):
    img = np.zeros((IMG_SIZE, IMG_SIZE))
    mean_iou = 0
    for el in annotation:
        draw_one_box(img, el[1], el[2], el[3], el[4])
        iou = calc_iou(img)
        mean_iou += iou / len(annotation)
        img[:, :] = 0
    return mean_iou


def test():
    path_fo_ds = Path('/home/max/projects/fiftyone/ds_sar')
    source_folder = Path('/home/max/projects/dataset_v2/processing/ships_data_big')
    labels_folder = path_fo_ds / 'labels'
    splits = ['train', 'val']
    ious = []
    mean_ious = []
    img2iou = {}
    for split in tqdm(splits):
        for f_name in tqdm(os.listdir(labels_folder / split)):
            annotation = np.loadtxt(labels_folder / split / f_name, ndmin=2)
            #stencil_image_path = source_folder / f_name.replace('.txt', '') / 'image.png'
            #img = plt.imread(stencil_image_path)
            iou = calc_all_iou(annotation)
            ious.append(iou)

            mean_iou = calc_mean_iou(annotation)
            mean_ious.append(mean_iou)

            img2iou[f_name.replace('.txt', '')] = {'img_path': str(source_folder / f_name.replace('.txt', '') / 'image.png'),
                                                   'image_name': f_name.replace('.txt', ''),
                                                   'iou': iou, 'annotattion': annotation.tolist(), 'mean_iou': mean_iou}


    with open('scripts/img2iou.json', 'w') as file:
        json.dump(img2iou, file)
    np.save('scripts/ious.npy', ious)
    np.save('scripts/mean_ious.npy', mean_ious)
    plt.hist(ious, bins=200)
    plt.ylim(0, 100)
    plt.show()
            # plt.imshow(img, cmap='gray')
            # plt.show()
            # break


test()