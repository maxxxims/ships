from pathlib import Path
import numpy as np
from tqdm import tqdm
from system_executor import Executor
import matplotlib.pyplot as plt
import cv2

from generate_coods.draw_ships import open_json_file, get_image_annotation, make_annotation_for_all_ships, save_result, get_one_ship_indexes
from generate_coods.make_coords import convert_to_coords, save_coords


def draw_ships(annotations):
    print(annotations[0]['bbox'])
    img = np.zeros(shape=(800, 800, 3))
    for ann in annotations:
        x_indexes, y_indexes = get_one_ship_indexes(img, ann)
        img[x_indexes, y_indexes] = [1, 1, 1]
    plt.imshow(img)
    plt.show()
    return img


def make_bbox_annotation(save_folder: Path, file_name: Path = Path('P0137_92.jpg')):
    IMG_SIZE = 1280
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


    def convert_coord(x: float, y: float, as_int: bool = False):
        x_new = x - (1760 - 640)
        y_new = y - (1237 - 640)
        if as_int:
            return (int(x_new), int(y_new))
        return (x_new, y_new)

    def to_bbox_data(x_bbox_indexes, y_bbox_indexes):
        delta_d = 0.12
        delta_a = 0.495
        x_bbox_indexes = [delta_d * el + 1760 for el in x_bbox_indexes]
        y_bbox_indexes = [1237 - delta_a * el for el in y_bbox_indexes]
        return x_bbox_indexes, y_bbox_indexes
    

    img2 = draw_ships(annotations)

    bbox_coords = []
    for ann in result:
        x_bbox_indexes, y_bbox_indexes = to_bbox_data(ann['x_indexes'] - 400, ann['y_indexes'] - 400)
        x_left = np.min(x_bbox_indexes) - 5
        x_right = np.max(x_bbox_indexes) + 5
        y_left = np.min(y_bbox_indexes) - 5
        y_right = np.max(y_bbox_indexes) + 5



        x_left, y_left = convert_coord(x_left, y_left, as_int=True)
        x_right, y_right = convert_coord(x_right, y_right, as_int=True)
        
        print(f'calculated = {x_left}, {y_left}, {x_right}, {y_right}')
        print(f'real = {ann["x_bbox_indexes"]}, {ann["y_bbox_indexes"]}')

        img2 = cv2.rectangle(img2, pt1=(x_left, y_left), pt2=(x_right, y_right), color=(1, 0, 0), thickness=1)

        x_center = (x_left + x_right) / 2
        y_center = (y_left + y_right) / 2
        h = (x_right - x_left) / IMG_SIZE
        w = (y_right - y_left) / IMG_SIZE
        x_center = x_center / IMG_SIZE
        y_center = y_center / IMG_SIZE

        bbox_coords.append(f'0 {x_center} {y_center} {h} {w}')

    plt.imshow(img2)
    plt.show()
    # return x_center, y_center, h, w
    # with open(save_folder / 'annotation_custon.txt', 'w') as file:
    #     for string in bbox_coords:
    #         file.write(f'{string}\n')




def make_bbox_annotation_for_all(folder: Path):
    for file_name in tqdm(folder.iterdir()):
        if '.' not in file_name.name:
            image_name = f'{file_name.name}.jpg'
            save_folder = file_name
            make_bbox_annotation(save_folder, file_name=image_name)
            break


if __name__ == "__main__":
    path = Executor.main_path / Executor.PROCESSING / 'ships_data'
    make_bbox_annotation_for_all(path)