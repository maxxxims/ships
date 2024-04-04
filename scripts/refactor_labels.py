from pathlib import Path
import os
import shutil
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import json
from sklearn.model_selection import train_test_split



SPLITS = ['val', 'train']



def merge_one_folder(path_to_ds: Path, splits: list = None):
    if splits is None:  splits = SPLITS

    labels_folder = path_to_ds / 'labels'
    images_folder = path_to_ds / 'images'

    tmp_labels_path = path_to_ds / 'tmp_labels'
    tmp_images_path = path_to_ds / 'tmp_images'

    tmp_labels_path.mkdir(exist_ok=True, parents=True)
    tmp_images_path.mkdir(exist_ok=True, parents=True)

    exists_tmp_labels_names = os.listdir(tmp_labels_path)

    for split in splits:
        for label_name in tqdm(os.listdir(labels_folder / split)):
            image_name = label_name.replace('.txt', '.png')
            if label_name not in exists_tmp_labels_names:
                shutil.move(labels_folder / split / label_name, tmp_labels_path / label_name)
                shutil.move(images_folder / split / image_name, tmp_images_path / image_name)


def make_new_ds_split(path_to_ds, good_samples: list, random_state: int = 42, test_size: int = 0.2):
    tmp_labels_path = path_to_ds / 'tmp_labels'
    tmp_images_path = path_to_ds / 'tmp_images'
    assert tmp_labels_path.exists() and tmp_images_path.exists(), "tmp folders don't exist! run merge_one_folder() first"
    if good_samples is not None:
        good_labels = []
        for label_name in tqdm(os.listdir(tmp_labels_path)):
            sample_name = label_name.replace('.txt', '')
            if sample_name in good_samples:
                good_labels.append(sample_name)
    else:
        good_labels = [el.replace('.txt', '') for el in os.listdir(tmp_labels_path)]
    
    train_labels, val_labels = train_test_split(good_labels, test_size=test_size, random_state=random_state)

    for sample_name in train_labels:
        lable_name = sample_name + '.txt'
        image_name = sample_name + '.png'
        shutil.move(tmp_labels_path / lable_name, path_to_ds / 'labels' / 'train' / lable_name)
        shutil.move(tmp_images_path / image_name, path_to_ds / 'images' / 'train' / image_name)
    

    for sample_name in val_labels:
        lable_name = sample_name + '.txt'
        image_name = sample_name + '.png'
        shutil.move(tmp_labels_path / lable_name, path_to_ds / 'labels' / 'val' / lable_name)
        shutil.move(tmp_images_path / image_name, path_to_ds / 'images' / 'val' / image_name)

def get_ds_info(path_to_ds: Path):
    labels_folder = path_to_ds / 'labels'
    images_folder = path_to_ds / 'images'
    print(f"""
    val: images = {len(os.listdir(images_folder / 'val'))}, labels = {len(os.listdir(labels_folder / 'val'))}
    train: images = {len(os.listdir(images_folder / 'train'))}, labels = {len(os.listdir(labels_folder / 'train'))}
    """)
    print(f"all = {len(os.listdir(labels_folder / 'val')) + len(os.listdir(labels_folder / 'train'))}")


# def delete_bad_samples(path_to_ds: Path, good_samples: list):
#     tmp_labels_path = path_to_ds / 'tmp_labels'
#     tmp_images_path = path_to_ds / 'tmp_images'
#     assert tmp_labels_path.exists() and tmp_images_path.exists(), "tmp folders don't exist! run merge_one_folder() first"

#     for label_name in tqdm(os.listdir(tmp_labels_path)):
#         if label_name not in good_samples:
#             os.remove(tmp_labels_path / label_name)
#             os.remove(tmp_images_path / label_name.replace('.txt', '.png'))
    

if __name__ == "__main__":
    path_to_ds = Path('/home/max/projects/fiftyone/large2/ds_compressed')
    image_names_path = Path('scripts/image_names.json')
    image_names = json.load(open(image_names_path, 'r'))[0]['images']
    # print(image_names)
    merge_one_folder(path_to_ds)
    make_new_ds_split(path_to_ds, image_names)
    get_ds_info(path_to_ds)
    ...