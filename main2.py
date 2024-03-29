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
from generate_coods.validation import has_close_ships


"""
'center_line': 1237, 'center_col': 1760,
'line_radius': 640, 'col_radius': 640
"""


def draw_ships_(save_folder: Path, transformer: AnnotationTransformer, result: list, scale_factor: float = 1):
    
    find_contours = 0

    #img_orig = plt.imread(save_folder / 'image.png')
    for r in result:
        x_indexes, y_indexes, z_indexes = r.get('x_indexes'), r.get('y_indexes'), r.get('z_indexes')
        result_coords = transformer.transform_coords(x_indexes, y_indexes, z_indexes)
        img = np.zeros((1280, 1280, 3), dtype=np.uint8)
        for el in result_coords:
            lineI, colI = el
            lineI = round(lineI - (1237 - 640))
            colI = round(colI - (1760 - 640))
            img[lineI, colI] = [255, 0, 0]
            #img_orig[lineI, colI] = [1, 0, 1, 1]

        #set a thresh
        thresh = 10
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #get threshold image
        ret,thresh_img = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
        #find contours
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #create an empty image for contours
        IMG_SIZE = 1280
        with open(save_folder / 'annotation_corrected2.txt', 'a+') as file:
            for cnt in contours:
                find_contours += 1
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
                #img_orig = cv2.rectangle(img_orig,(x_left,y_left),(x_right,y_right),(1,1,1),1)
                #img_orig = cv2.rectangle(img_orig,(x_left,y_left),(x_right,y_right),(255,255,255),1)
                #plt.imshow(img, cmap='gray')
                #plt.show()
        #rect = cv2.minAreaRect(cnt) # пытаемся вписать прямоугольник
        #box = cv2.boxPoints(rect) # поиск четырех вершин прямоугольника
        #box = np.int0(box) # округление координат
        #img_contours = cv2.drawContours(img_contours, [box], 0, (255,255,255), 1)
    #cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
    #print(f'find_contours = {find_contours} / {len(result)}')
    #plt.imshow(img_orig, cmap='gray')
    #plt.show()
    #plt.imsave(save_folder / 'image_cutted.png', img_orig)
    ...



def make_annotation_for_image(path_to_folder: Path, scale_factor: float = 1):
    """
        make bbox by cv2 and save
    """
    #img = np.copy(plt.imread(path_to_folder / 'image.png'))
    img_gray = cv2.imread(str(path_to_folder / 'image.png'), cv2.IMREAD_GRAYSCALE)
    thresh = 30
    #img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    plt.imshow(img_gray, cmap='gray')
    plt.show()


    ret,thresh_img = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    plt.imshow(thresh_img, cmap='gray')
    plt.show()

    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #img_contours = np.uint8(np.zeros((img.shape[0],img.shape[1])))
    IMG_SIZE = 1280
    with open(path_to_folder / 'annotation_corrected4.txt', 'w') as file:
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            x_center, y_center = x + w / 2, y + h / 2
            w, h = w * scale_factor, h * scale_factor
            x_left = round(x_center - w / 2)
            y_left = round(y_center - h / 2)
            x_right = round(x_center + w / 2)
            y_right = round(y_center + h / 2)

            img_gray = cv2.rectangle(img_gray,(x_left,y_left),(x_right,y_right),(255,255,255),1)

            x_center, y_center = x_center / IMG_SIZE, y_center / IMG_SIZE
            w, h = w / IMG_SIZE, h / IMG_SIZE
            file.write(f'0 {x_center} {y_center} {w} {h}\n')
        
        #rect = cv2.minAreaRect(cnt) # пытаемся вписать прямоугольник
        #box = cv2.boxPoints(rect) # поиск четырех вершин прямоугольника
        #box = np.int0(box) # округление координат
        #img_contours = cv2.drawContours(img_contours, [box], 0, (255,255,255), 1)
    #cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
    plt.imshow(img_gray, cmap='gray')
    plt.show()


def draw_annotations(save_folder: Path,annotations: list):
    img = np.zeros(shape=(800, 800))
    for ann in annotations:
        full_len = len(ann['segmentation'][0])
        segment = (np.reshape(ann['segmentation'][0], newshape=(full_len//2, 2))).astype(np.int32)
        img = cv2.fillPoly(img, [segment], 255)
    plt.imsave(save_folder / 'annotations.png', img)



def make_config_for_image(save_folder: Path, file_name: str = 'P0137_92.jpg'):
    path_to_labels = Path('draw/labels.json')
    save_temporary_result_path = Path('draw/result.json')
    save_path = Path('/home/max/projects/dataset_v2/simulation/group_0_data.json')  # path to json file with coords for generator

    annotation_data = open_json_file(path_to_labels)   # dict with special data for image annotation in dataset
    image, annotations = get_image_annotation(annotation_data, file_name=file_name)   #P0001_1200_2000_8400_9200.jpg
    #print(f'file_name = {image["file_name"]}; and from func = {file_name}')
    
    #has_close_ships(annotations)
    draw_annotations(save_folder, annotations)


    img = np.zeros(shape=(800, 800, 3))
    result = make_annotation_for_all_ships(img, annotations, new_points=5, scale_img=False)

    #assert len(result) == len(annotations), f'len(result) = {len(result)} != len(annotations) = {len(annotations)}'

    save_result(image['file_name'], result, save_result_path=save_temporary_result_path)

    transformer = AnnotationTransformer(path_to_params=Executor.main_path / Executor.SUMULATION_PARAMS)
    save_annotation_v2(result, save_folder, transformer)
    save_annotation(result, save_folder)


    image_index_data = open_json_file(save_temporary_result_path)
    coords = convert_to_coords(image_index_data, limit = None)
    save_coords(coords, save_path=save_path)
    draw_ships_(save_folder, transformer, result)


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
        #make_annotation_for_image(path_to_folder=folder, scale_factor=1)
        #Executor.save_annotation(scale_factor=2)
        # Executor.show_gologram(cutted=True)
        #Executor.show_image(cutted=True, show_annotation=True)

    else: 
        Executor.set_save_dir(folder)
        
        Executor.show_image(cutted=False)
        Executor.draw_range_compressed(cutted=False)
        Executor.show_gologram(cutted=False)


def make_dataset(i_start, i_end):
    ds_path = Path('/home/max/projects/dataset_v2/processing/ships_data_big')
    ds_path.mkdir(exist_ok=True, parents=True)

    processed_txt_path = ds_path / 'processed.txt'
    with open(processed_txt_path, 'w') as file:
        for el in os.listdir(ds_path):
            if '.' not in el:
                file.write(f"{el} -1" + '\n')

    path_to_labels = Path('draw/labels.json')
    annotation_data = open_json_file(path_to_labels)
    #print(f'len of images = {len(annotation_data["images"])}')
    with open(processed_txt_path, 'a+') as file:
        for i in trange(i_start, i_end):
            file_name = annotation_data['images'][i]['file_name']
            folder = ds_path / file_name.replace('.jpg', '')
            folder.mkdir(exist_ok=True, parents=True)

            make_config_for_image(file_name=file_name, save_folder=folder)
            run_cmd(folder, generate=True)
            
            file.write(f"{str(file_name.replace('.jpg', ''))} {i}" + '\n')


def to_one_folder():
    new_folder = Executor.main_path / 'IMAGES3'
    new_folder.mkdir(exist_ok=True)
    annotation_folder = new_folder / 'labels' / 'val'
    annotation_folder.mkdir(exist_ok=True, parents=True)
    images_folder = new_folder / 'images' / 'val'
    images_folder.mkdir(exist_ok=True, parents=True)


    folder = Executor.main_path / Executor.PROCESSING / 'ships_data2'
    diff = 0
    for el in tqdm(folder.iterdir()):
        if not '.' in el.name:
            real = np.loadtxt(folder / el / 'annotation.txt', ndmin=2)
            v2 = np.loadtxt(folder / el / 'annotation_corrected2.txt', ndmin=2)
            if len(real) - len(v2) > 2:
                diff += 1
            shutil.copy(folder / el / 'image.png', images_folder / f'{el.name}.png')
            shutil.copy(folder / el / 'annotation_corrected2.txt', annotation_folder / f'{el.name}.txt')
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
    # to_one_folder()
    make_dataset(0, 5602)
    #make_config_for_image(file_name=file_name, save_folder=save_folder_test)
    # run_cmd(save_folder_test, generate=True)
    
