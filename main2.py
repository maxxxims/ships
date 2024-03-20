from pathlib import Path
import shutil
from generate_coods.draw_ships import open_json_file, get_image_annotation, make_annotation_for_all_ships, save_result, get_one_ship_indexes
from generate_coods.make_coords import convert_to_coords, save_coords
from generate_coods.make_annotation import save_annotation, save_annotation_v2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from system_executor import Executor
from tqdm import  trange, tqdm
import cv2
from make_annotation import AnnotationTransformer




def make_config_for_image(save_folder: Path, file_name: str = 'P0137_92.jpg'):
    path_to_labels = Path('draw/labels.json')
    save_temporary_result_path = Path('draw/result.json')
    save_path = Path('/home/max/projects/dataset_v2/simulation/group_0_data.json')  # path to json file with coords for generator

    annotation_data = open_json_file(path_to_labels)   # dict with special data for image annotation in dataset
    image, annotations = get_image_annotation(annotation_data, file_name=file_name)   #P0001_1200_2000_8400_9200.jpg
    #print(f'file_name = {image["file_name"]}; and from func = {file_name}')
    
    img = np.zeros(shape=(800, 800, 3))
    result = make_annotation_for_all_ships(img, annotations, new_points=60, scale_img=False)

    save_result(image['file_name'], result, save_result_path=save_temporary_result_path)

    transformer = AnnotationTransformer(path_to_params=Executor.main_path / Executor.SUMULATION_PARAMS)
    save_annotation_v2(result, save_folder, transformer)


    image_index_data = open_json_file(save_temporary_result_path)
    coords = convert_to_coords(image_index_data, limit = None)
    save_coords(coords, save_path=save_path)


def run_cmd(folder: Path, generate=False):
    if generate:
        Executor.set_save_dir(folder)
        Executor.generate_output_signal()
        Executor.generate_hologram()
        Executor.range_compress()
        Executor.move_result_file_to_save_folder()

        Executor.cut_comressed_hologram(save_png=True)
        Executor.cut_hologram(save_png=True)
        Executor.cut_image(save_png=True, cut=True)
        # Executor.show_gologram(cutted=True)
        Executor.show_image(cutted=True, show_annotation=True)

    else: 
        Executor.set_save_dir(folder)
        
        Executor.show_image(cutted=False)
        Executor.draw_range_compressed(cutted=False)
        Executor.show_gologram(cutted=False)


if __name__ == '__main__':
    img = np.zeros((300, 600), dtype=np.uint8)
    #img = img[0:100, 0:500] #lines, cols
    file_name = 'P0002_5260_6060_6000_6800.jpg'


    save_folder_test = Executor.main_path / 'test_data_ann'
    save_folder_test.mkdir(exist_ok=True,parents=True)

    make_config_for_image(file_name=file_name, save_folder=save_folder_test)
    run_cmd(save_folder_test, generate=True)
