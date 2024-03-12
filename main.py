from pathlib import Path
from generate_coods.draw_ships import open_json_file, get_image_annotation, make_annotation_for_all_ships, save_result, get_one_ship_indexes
from generate_coods.make_coords import convert_to_coords, save_coords
from generate_coods.make_annotation import save_annotation
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from system_executor import Executor
from tqdm import  trange, tqdm
import cv2


def draw_ships(annotations, result = None, save_folder = None):
    print(annotations[0]['bbox'])
    img = np.zeros(shape=(800, 800, 3))
    for ann in annotations:
        x_indexes, y_indexes = get_one_ship_indexes(img, ann)
        img[x_indexes, y_indexes] = [1, 1, 1]
        #x_left, y_left, h, w = ann['bbox']
        #pt1 = (int(x_left), int(y_left))
        #pt2 = (int(x_left+h), int(y_left +w))
        #img = cv2.rectangle(img, pt1=pt1, pt2=pt2, color=(1, 0, 0), thickness=1)
    if save_folder is not None:
        plt.imsave(save_folder / 'image_annotated.png', img)
    else:
        plt.imshow(img)
        plt.show()


def make_config_for_image(save_folder: Path, file_name: str = 'P0137_92.jpg'):
    path_to_labels = Path('draw/labels.json')
    path_to_image_folder = Path('draw')
    save_temporary_result_path = Path('draw/result.json')
    save_path = Path('/home/max/projects/dataset_v2/simulation/group_0_data.json')  # path to json file with coords for generator

    annotation_data = open_json_file(path_to_labels)   # dict with special data for image annotation in dataset
    # exit(0)
    image, annotations = get_image_annotation(annotation_data, file_name=file_name)   #P0001_1200_2000_8400_9200.jpg
    #print(f'file_name = {image["file_name"]}; and from func = {file_name}')
    
    img = np.zeros(shape=(800, 800, 3))
    result = make_annotation_for_all_ships(img, annotations, new_points=60, scale_img=False)
    save_annotation(result, save_folder)
    #draw_ships(annotations, save_folder=save_folder)
    # exit(0)
    save_result(image['file_name'], result, save_result_path=save_temporary_result_path)


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
        Executor.cut_image(save_png=True)
        # Executor.show_gologram(cutted=True)
        Executor.show_image(cutted=True,show_annotation=True)

    else: 
        Executor.set_save_dir(folder)
        
        Executor.show_image(cutted=False)
        Executor.draw_range_compressed(cutted=False)
        Executor.show_gologram(cutted=False)
        


def get_random_image():
    path_to_labels = Path('draw/labels.json')
    annotation_data = open_json_file(path_to_labels)
    print(annotation_data['images'])
    file_name ='P0001_4800_5600_6600_7400.jpg'
    image, annotations = get_image_annotation(annotation_data, file_name=file_name)
    draw_ships(annotations)


def get_image_name(idx: int):
    path_to_labels = Path('draw/labels.json')
    annotation_data = open_json_file(path_to_labels)
    # print(annotation_data['images'][0])
    # print(annotation_data['annotations'][0])
    return annotation_data['images'][idx]['file_name']


def make_data(i_start, i_end):
    save_folder = Path('ships_data')
    # del previous data
    processed_txt_path = Executor.main_path / Executor.PROCESSING / save_folder / 'processed.txt'
    with open(processed_txt_path, 'w') as file:
        for el in os.listdir(Executor.main_path / Executor.PROCESSING / save_folder):
            if '.' not in el:
                file.write(f"{el} -1" + '\n')

    for i in trange(i_start, i_end):
        file_name = get_image_name(i)
        folder = save_folder / file_name.replace('.jpg', '')
        full_save_folder = Executor.main_path / Executor.PROCESSING / folder
        full_save_folder.mkdir(exist_ok=True, parents=True)
        make_config_for_image(file_name=file_name, save_folder=full_save_folder)
        run_cmd(folder, generate=True)
        with open(processed_txt_path, 'a+') as file:
            file.write(f"{str(file_name.replace('.jpg', ''))} {i}" + '\n')


if __name__ == "__main__":
    file_name = 'P0002_5260_6060_6000_6800.jpg'
    # make_config_for_image(file_name=file_name)
    #folder = Path(file_name.replace('.jpg', ''))  #Path('ships_5_moved_60')
    #run_cmd(Path('test_data'), generate=False)
    #get_random_image()
    # get_image_name(0)
    make_data(0, 1000)

    # сохранять png
    # разметка
    # сделать интерфейс