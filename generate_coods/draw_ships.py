import json
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from transformation import CoordinateConversion

"""""""""
    This script get annotations from dataset and convert it to polygons of ships 
    and also add new points between close ones
"""""""""



MASK_COLOR = [255, 0, 0]
DEFAULT_COLOR = [0, 255, 255]
reference_point = [0.0, 431785.69, 6385417.85]


def open_json_file(path: Path) -> dict:
    with open(path, 'r') as file:
        return json.load(file)


def get_image_annotation(data: dict, file_name: str):
    """
        return annotations for one image from labels.json
    """
    for image in data['images']:
        if image['file_name'] == file_name:
            break
    id = image['id']
    print(
        f'id = {id}; name = {image["file_name"]}'
    )
    annotations = []
    for annotation in data['annotations']:
        if annotation['image_id'] == id:
            annotations.append(annotation)
    return image, annotations


def get_one_ship_indexes(img: np.ndarray, annotation: dict):
    """
        return indexes on image with ship
    """
    full_len = len(annotation['segmentation'][0])
    segment = (np.reshape(annotation['segmentation'][0], newshape=(full_len//2, 2))).astype(np.int32)
    new_img = cv2.fillPoly(img, [segment], MASK_COLOR)
    x_indexes, y_indexes = np.where((new_img[:, :, 0] == MASK_COLOR[0]) & (new_img[:, :, 1] == MASK_COLOR[1]) & (new_img[:, :, 2] == MASK_COLOR[2]))
    return x_indexes, y_indexes


def move_points(x_indexes, y_indexes, x_move = 400, y_move = 400):
    return np.array(x_indexes) - x_move, np.array(y_indexes) - y_move



def get_one_ship_indexes_scale(img: np.ndarray, annotation: dict):
    """
        return indexes on scaled image with ship
    """
    full_len = len(annotation['segmentation'][0])
    segment = (np.reshape(annotation['segmentation'][0], newshape=(full_len//2, 2))).astype(np.int32)
    new_img = cv2.fillPoly(img, [segment], MASK_COLOR)
    x_scale, y_scale = 4, 1
    new_img = cv2.resize(new_img, (new_img.shape[0] * x_scale, new_img.shape[1] * y_scale))
    x_indexes, y_indexes = np.where((new_img[:, :, 0] == MASK_COLOR[0]) & (new_img[:, :, 1] == MASK_COLOR[1]) & (new_img[:, :, 2] == MASK_COLOR[2]))
    return x_indexes, y_indexes


def make_annotation_for_all_ships(img: np.ndarray, annotations, new_points: int = 1, scale_img: bool = False):
    """
        make result file with indexes of all ships on image
    """
    result = []
    transformer = CoordinateConversion(reference_point[0], reference_point[1], reference_point[2], 6400000.0)
    for i, annotation in enumerate(annotations):
        if scale_img:
            x_indexes, y_indexes = get_one_ship_indexes_scale(np.copy(img), annotation)
        else:
            y_indexes, x_indexes  = get_one_ship_indexes(np.copy(img), annotation)
        
        #img[x_indexes, y_indexes] = 255

        x_indexes, y_indexes = add_more_points(x_indexes, y_indexes, new_points=new_points)
        # MOVE IMAGE 400 at every axis!!!
        x_indexes, y_indexes = move_points(x_indexes, y_indexes, x_move=400, y_move=400)
        x_indexes_new, y_indexes_new, z_indexes_new = transformer.transform_local_image_to_GPDSK(x_indexes, y_indexes)

        #ADD BBOX
        x_left, y_left, h, w = annotation['bbox']
        x_right, y_right = x_left + h, y_left + w

        ##img = cv2.rectangle(img, (int(x_left), int(y_left)), (int(x_right), int(y_right)), 255, 1)
        ##img[int(y_left):int(y_right), int(x_left):int(x_right)] += 100
        
        
        x_bbox_indexes, y_bbox_indexes = np.array([x_left, x_right]), np.array([y_left, y_right])
        x_bbox_indexes, y_bbox_indexes = move_points(x_bbox_indexes, y_bbox_indexes, x_move=400, y_move=400)
        x_bbox_indexes, y_bbox_indexes, z_bbox_indexes = transformer.transform_local_image_to_GPDSK(x_bbox_indexes, y_bbox_indexes)
        ##delta_d = 0.12
        ##delta_a = 0.495
        ##x_bbox_indexes = [delta_d * el + 1760 for el in x_bbox_indexes]
        ##y_bbox_indexes = [1237 - delta_a * el for el in y_bbox_indexes]


        result.append({
            'index': i,
            'x_indexes': x_indexes_new,
            'y_indexes': y_indexes_new,
            'z_indexes': z_indexes_new,
            'x_bbox_indexes': x_bbox_indexes,
            'y_bbox_indexes': y_bbox_indexes,
            'z_bbox_indexes': z_bbox_indexes

            # 'x_indexes_original':  x_indexes,
            # 'y_indexes_original':  y_indexes,
        })
    ##img[img > 255] = 255
    ##plt.imshow(img)
    ##plt.show()
    return result


def add_more_points(x_indexes: list, y_indexes: list, new_points: int = 1) -> tuple:
    """
        add extra points between close one's
    """
    #print(f'NEW POINTS = {new_points}')
    new_x_indexes, new_y_indexes = [], []
    for i in range(len(x_indexes) - 1):
        x, y = x_indexes[i], y_indexes[i]
        new_x_indexes.append(x)
        new_y_indexes.append(y)
        for j in range(new_points):
            new_x_coord = x + ((x_indexes[i+1] - x) * ((1 + j) / (1 + new_points)))
            new_y_coord = y + ((y_indexes[i+1] - y) * ((1 + j) / (1 + new_points)))
            new_x_indexes.append(new_x_coord)
            new_y_indexes.append(new_y_coord)
    new_x_indexes.append(x_indexes[-1])
    new_y_indexes.append(y_indexes[-1])
    return new_x_indexes, new_y_indexes



def show_annotation(img, annotation: dict):
    x_indexes, y_indexes = np.array(annotation.get('x_indexes'), dtype=int), np.array(annotation.get('y_indexes'), dtype=int)
    img[x_indexes, y_indexes] = DEFAULT_COLOR
    return img



def save_result(img_name: str,
                result: list,
                save_result_path: Path = Path('draw/result.json')):
    """
        save temporary results with indexes of points
    """

    data = {
        'annotation_number': len(result),
        'image_name': img_name,
        'original_annotations': [
            {
                'index': int(r['index']),
                'x_indexes': [float(el) for el in r['x_indexes']],
                'y_indexes': [float(el) for el in r['y_indexes']],
                'z_indexes': [float(el) for el in r['z_indexes']],

                # 'x_indexes_original': [float(el) for el in r['x_indexes_original']],
                # 'y_indexes_original': [float(el) for el in r['y_indexes_original']],
            }
            for r in result
        ],
    }
    

    with open(save_result_path, 'w') as file:
        json.dump(data, file)


if __name__ == '__main__':
    path = Path('draw/labels.json')
    path_to_images = Path('draw')
    data = open_json_file(path)

    image, annotations = get_image_annotation(data, 'P0001_1200_2000_8400_9200.jpg')

    img = np.copy(plt.imread(path_to_images / image['file_name']))
    img_copy = np.copy(img)

    result = make_annotation_for_all_ships(img, annotations, new_points=20)
    
    save_result(image['file_name'], result)