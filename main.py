from pathlib import Path
from draw_ships import open_json_file, get_image_annotation, make_annotation_for_all_ships, save_result, get_one_ship_indexes
from make_coords import convert_to_coords, save_coords
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from system_executor import Executor



def draw_ships(annotations):
    img = np.zeros(shape=(800, 800, 3))
    for ann in annotations:
        x_indexes, y_indexes = get_one_ship_indexes(img, ann)
        img[x_indexes, y_indexes] = [255, 255, 255]
    plt.imshow(img)
    plt.show()


def make_config_for_image(file_name: str = 'P0137_92.jpg'):
    path_to_labels = Path('draw/labels.json')
    path_to_image_folder = Path('draw')
    save_temporary_result_path = Path('draw/result.json')
    save_path = Path('/home/max/projects/dataset_v2/simulation/group_0_data.json')  # path to json file with coords for generator

    annotation_data = open_json_file(path_to_labels)   # dict with special data for image annotation in dataset
    # exit(0)
    image, annotations = get_image_annotation(annotation_data, file_name=file_name)   #P0001_1200_2000_8400_9200.jpg
    
    # draw_ships(annotations)
    # exit(0)
    img = np.zeros(shape=(800, 800, 3))
    result = make_annotation_for_all_ships(img, annotations, new_points=40, scale_img=False)
    save_result(image['file_name'], result, save_result_path=save_temporary_result_path)


    image_index_data = open_json_file(save_temporary_result_path)
    coords = convert_to_coords(image_index_data, limit = None)
    save_coords(coords, save_path=save_path)


def run_cmd(generate=False):
    if generate:
        Executor.generate_hologram()
        Executor.range_compress()
        Executor.draw_hologram()
        Executor.draw_image()
        Executor.draw_range_compressed()
    else: 
        Executor.set_save_dir(Path('ships_1'))
        Executor.draw_range_compressed()
        Executor.show_gologram()
        Executor.show_image()


def get_random_image():
    path_to_labels = Path('draw/labels.json')
    annotation_data = open_json_file(path_to_labels)
    print(annotation_data['images'])
    file_name ='P0001_4800_5600_6600_7400.jpg'
    image, annotations = get_image_annotation(annotation_data, file_name=file_name)
    draw_ships(annotations)


if __name__ == "__main__":
    file_name = 'P0002_4800_5600_6600_7400.jpg'
    make_config_for_image(file_name=file_name)
    run_cmd(generate=True)
    #get_random_image()

    