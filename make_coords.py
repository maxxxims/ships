import json
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict 


def open_json_file(path: Path):
    with open(path, 'r') as file:
        return json.load(file)
    

def prerpocess_coords(arr: list, c_type: str):
    if c_type == 'x':
        new_arr = np.array(arr) * 2
        return new_arr.tolist()
    elif c_type == 'y':
        new_arr = np.array(arr) * 6
        return new_arr.tolist()
    raise Exception("ERROR")


def convert_to_coords(data, limit = None) -> dict:
    """
        convert temporary results to special format for generator
    """
    if limit is None:   limit = data['annotation_number']
    coords = defaultdict(list)

    
    for i in range(limit):
        annotation = data.get('original_annotations')[i]
        # coords["x"] += prerpocess_coords(annotation['x_indexes'], c_type='x')
        # coords["y"] += prerpocess_coords(annotation['y_indexes'], c_type='y')
        coords["x"] += annotation['x_indexes']
        coords["y"] += annotation['y_indexes']
        coords["z"] += annotation['z_indexes']
        coords["nx"] += [0.0] * len(annotation['x_indexes']) 
        coords["ny"] += [0.0] * len(annotation['x_indexes']) 
        coords["nz"] += [1.0] * len(annotation['x_indexes']) 

    return coords


def verify_img():
    path = Path('draw/result_scale.json')
    data = open_json_file(path)
    img = np.zeros((6 * 800, 3 * 800))

    for el in data['original_annotations']:
        print(el['index'])
        img[np.array(el['x_indexes'], dtype=int), np.array(el['y_indexes'], dtype=int)] = 255
        # print(el['x_indexes'])
    plt.imshow(img)
    plt.show()


def make_test_coords():
    coords = defaultdict(list)
    coords['x'] += [0.0, 20.0, 40.0]
    coords['y'] += [0.0, 20.0, 40.0]
    coords['z'] += [0.0] * len(coords['x'])
    coords['nx'] += [0.0] * len(coords['x'])
    coords['ny'] += [0.0] * len(coords['x'])
    coords['nz'] += [1.0] * len(coords['x'])

    return coords

def save_coords(coords, save_path: Path):
    """
        save coords in json in simulation folder
    """
    with open(save_path, 'w') as file:
        json.dump(coords, file)
    print('done!')


def test_img_ship_huge_scale():
    path = Path('draw/result2.json')
    data = open_json_file(path)
    img = np.zeros((5000, 2500), dtype=np.uint8)
    coords = convert_to_coords(data, limit = None)

    img[np.array(coords['x'], dtype=int), np.array(coords['y'], dtype=int)] = 255
    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    # verify_img()
    # test_img_ship_huge_scale()
    # coords = make_test_coords()
    path = Path('draw/result.json')
    save_path = Path('/home/max/projects/dataset_v2/simulation/group_0_data.json')
    data = open_json_file(path)
    coords = convert_to_coords(data, limit = None)
    
    save_coords(coords, save_path=save_path)