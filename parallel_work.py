import os
from pathlib import Path
import shutil

from matplotlib import pyplot as plt
import numpy as np
from system_executor import Executor
from tqdm import tqdm
from sklearn.model_selection import train_test_split



def calc_bad_annotations():
    path_fo_ds = Path('/home/max/projects/dataset_v2/processing/ships_data2')
    folders = [f for f in os.listdir(path_fo_ds) if os.path.isdir(path_fo_ds / f)]
    diff = 0

    for folder in tqdm(folders):
        folder_files = os.listdir(path_fo_ds / folder)
        if 'annotation.txt' in folder_files and 'annotation_corrected2.txt' in folder_files:
            real = np.loadtxt(path_fo_ds / folder / 'annotation.txt', ndmin=2)
            corrected = np.loadtxt(path_fo_ds / folder / 'annotation_corrected2.txt', ndmin=2)
            if len(real) != len(corrected):
                diff += 1

    print(diff)


def copy_ds():
    path_fo_ds = Path('/home/max/projects/dataset_v2/processing/ships_data_big')
    folders = [f for f in os.listdir(path_fo_ds) if os.path.isdir(path_fo_ds / f)]
    path_to_new_ds = Path('/home/max/projects/fiftyone/ds')
    if path_to_new_ds.exists():
        shutil.rmtree(path_to_new_ds)
    path_to_new_ds.mkdir(exist_ok=True, parents=True)

    annotation_folder = path_to_new_ds / 'labels' / 'val'
    annotation_folder.mkdir(exist_ok=True, parents=True)
    images_folder = path_to_new_ds / 'images' / 'val'
    images_folder.mkdir(exist_ok=True, parents=True)

    raw_folder = path_to_new_ds / 'raw' / 'val'
    comressed_folder = path_to_new_ds / 'compressed' / 'val'
    raw_folder.mkdir(exist_ok=True, parents=True)
    comressed_folder.mkdir(exist_ok=True, parents=True)


    for folder in tqdm(folders):
        folder_files = os.listdir(path_fo_ds / folder)
        if 'annotation.txt' in folder_files and 'annotation_corrected2.txt' in folder_files and 'gologram_range_compressed.png' in folder_files:
            # print('here!')
            shutil.copy(path_fo_ds / folder / 'image.png', images_folder / f'{folder}.png')
            shutil.copy(path_fo_ds / folder / 'annotation_corrected2.txt', annotation_folder / f'{folder}.txt')
            
            shutil.copy(path_fo_ds / folder / 'gologram.png', raw_folder / f'{folder}.png')
            shutil.copy(path_fo_ds / folder / 'gologram_range_compressed.png', comressed_folder / f'{folder}.png')


    yaml_path = Path('/home/max/projects/fiftyone/IMAGES/dataset.yaml')
    shutil.copy(yaml_path, path_to_new_ds / f'dataset.yaml')



def move_annotations():
    path_fo_ds = Path('/home/max/projects/fiftyone/ds')
    annotation_folder = path_fo_ds / 'labels' / 'val'
    images_folder = path_fo_ds / 'images' / 'val'

    for f_name in tqdm(os.listdir(annotation_folder)):
        annotations = np.loadtxt(annotation_folder / f_name, ndmin=2)
        with open(annotation_folder / f_name, 'w') as file:
            for el in annotations:
                yc_new = el[2] + 5 / 1280
                file.write(f'0 {el[1]} {yc_new} {el[3]} {el[4]}' + '\n')



def split_dataset():
    path_fo_ds = Path('/home/max/projects/fiftyone/ds')
    annotation_folder = path_fo_ds / 'labels' / 'val'
    images_folder = path_fo_ds / 'images' / 'val'

    labels = os.listdir(annotation_folder)

    train_labels, val_labels = train_test_split(labels, test_size=0.2, random_state=42)

    annotation_train = path_fo_ds / 'labels' / 'train'
    images_train = path_fo_ds / 'images' / 'train'
    annotation_train.mkdir(exist_ok=True, parents=True)
    images_train.mkdir(exist_ok=True, parents=True)


    raw_folder = path_fo_ds / 'raw' / 'val'
    comressed_folder = path_fo_ds / 'compressed' / 'val'
    raw_train = path_fo_ds / 'raw' / 'train'
    raw_train.mkdir(exist_ok=True, parents=True)
    compressed_train = path_fo_ds / 'compressed' / 'train'
    compressed_train.mkdir(exist_ok=True, parents=True)


    for f_name in tqdm(train_labels):
        shutil.move(annotation_folder / f_name, annotation_train / f_name)
        image_name = f_name.replace('.txt', '.png')
        shutil.move(images_folder / image_name, images_train / image_name)

        shutil.move(raw_folder / image_name, raw_train / image_name)
        shutil.move(comressed_folder / image_name, compressed_train / image_name)






if __name__ == '__main__':
    #copy_ds()
    #move_annotations()
    split_dataset()