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


"""
'center_line': 1237, 'center_col': 1760,
'line_radius': 640, 'col_radius': 640
"""


def draw_ships_(save_folder: Path, transformer: AnnotationTransformer, result: list, scale_factor: float = 2):
    # hsv_min = np.array((2, 28, 65), np.uint8)
    # hsv_max = np.array((26, 238, 255), np.uint8)
    # old_img = plt.imread(save_folder / 'image.png')
    img = np.zeros((1280, 1280, 3), dtype=np.uint8)
    print(img.shape)
    #img[:, :, -1] = 1
    for r in result:
        x_indexes, y_indexes, z_indexes = r.get('x_indexes'), r.get('y_indexes'), r.get('z_indexes')
        result_coords = transformer.transform_coords(x_indexes, y_indexes, z_indexes)

        for el in result_coords:
            lineI, colI = el
            lineI = round(lineI - (1237 - 640))
            colI = round(colI - (1760 - 640))
            img[lineI, colI] = [255, 0, 0]

    
    #set a thresh
    thresh = 10
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #get threshold image
    ret,thresh_img = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)

    #find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #create an empty image for contours
    img_contours = np.uint8(np.zeros((img.shape[0],img.shape[1])))
    IMG_SIZE = 1280
    with open(save_folder / 'annotation_corrected2.txt', 'w') as file:
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            x_center, y_center = x + w / 2, y + h / 2
            w, h = w * scale_factor, h * scale_factor
            x_left = round(x_center - w / 2)
            y_left = round(y_center - h / 2)
            x_right = round(x_center + w / 2)
            y_right = round(y_center + h / 2)

            x_center, y_center = x_center / IMG_SIZE, y_center / IMG_SIZE
            w, h = w / IMG_SIZE, h / IMG_SIZE

            file.write(f'0 {x_center} {y_center} {w} {h}\n')

        #img_contours = cv2.rectangle(img,(x_left,y_left),(x_right,y_right),(255,255,255),1)
        # rect = cv2.minAreaRect(cnt) # пытаемся вписать прямоугольник
        # box = cv2.boxPoints(rect) # поиск четырех вершин прямоугольника
        # box = np.int0(box) # округление координат
        # img_contours = cv2.drawContours(img_contours, [box], 0, (255,255,255), 1)
    #cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
    
    # plt.imshow(img_contours, cmap='gray')
    # plt.show()
    ...


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
    draw_ships_(save_folder, transformer, result)


def run_cmd(folder: Path, generate=False):
    if generate:
        Executor.set_save_dir(folder)
        #Executor.generate_output_signal()
        #Executor.generate_hologram()
        #Executor.range_compress()
        #Executor.move_result_file_to_save_folder()

        #Executor.cut_comressed_hologram(save_png=True)
        # Executor.cut_hologram(save_png=True)
        Executor.cut_image(save_png=True, cut=True)
        Executor.save_annotation(scale_factor=2)
        # Executor.show_gologram(cutted=True)
        #Executor.show_image(cutted=True, show_annotation=True)

    else: 
        Executor.set_save_dir(folder)
        
        Executor.show_image(cutted=False)
        Executor.draw_range_compressed(cutted=False)
        Executor.show_gologram(cutted=False)



def new_annotation():
    main_folder = Path('/home/max/projects/dataset_v2/processing/ships_data')
    for folder in main_folder.iterdir():
        if '.' not in folder.name:
            img_name = f'{folder.name}.jpg'
            make_config_for_image(save_folder=folder, file_name=img_name)
            #run_cmd(folder, generate=True)


def to_one_folder():
    new_folder = Executor.main_path / 'IMAGES2'
    new_folder.mkdir(exist_ok=True)
    annotation_folder = new_folder / 'labels' / 'val'
    annotation_folder.mkdir(exist_ok=True, parents=True)
    images_folder = new_folder / 'images' / 'val'
    images_folder.mkdir(exist_ok=True, parents=True)


    folder = Executor.main_path / Executor.PROCESSING / 'ships_data'
    diff = 0
    for el in tqdm(folder.iterdir()):
        if not '.' in el.name:
            real = np.loadtxt(folder / el / 'annotation_corrected.txt', ndmin=2)
            v2 = np.loadtxt(folder / el / 'annotation_corrected2.txt', ndmin=2)
            if len(real) - len(v2) > 2:
                diff += 1
            #shutil.copy(folder / el / 'image.png', images_folder / f'{el.name}.png')
            #shutil.copy(folder / el / 'annotation_corrected2.txt', annotation_folder / f'{el.name}.txt')
    print(diff)
    # bad P0001_1800_2600_10190_10990


def save_new_data():
    new_folder = Executor.main_path / 'SAVE_DATA'
    folder = Executor.main_path / Executor.PROCESSING / 'ships_data'
    for el in tqdm(folder.iterdir()):
        if not '.' in el.name:
            directory_new = new_folder / el.name
            directory_new.mkdir(exist_ok=True, parents=True)
            shutil.copy(folder / el / 'annotation_corrected2.txt', directory_new / 'annotation_corrected2.txt')
            shutil.copy(folder / el / 'annotation_corrected.txt', directory_new / 'annotation_corrected.txt')


if __name__ == '__main__':
    img = np.zeros((300, 600), dtype=np.uint8)
    #img = img[0:100, 0:500] #lines, cols
    file_name = 'P0002_5260_6060_6000_6800.jpg'


    save_folder_test = Executor.main_path / 'test_data_ann'
    save_folder_test.mkdir(exist_ok=True,parents=True)

    #make_config_for_image(file_name=file_name, save_folder=save_folder_test)
    # run_cmd(save_folder_test, generate=True)
    
    #new_annotation()
    #to_one_folder() #(a, b) - b - вертикаль 6 раз больше
    save_new_data()