import json
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


"""""""""
Thist script used to save anotation to save annotation
"""""""""



def save_annotation(result: list, save_folder: Path):
    """
        write bbox coords in full-scaled SAR image
    """
    # print(f'saved_folder = {save_folder}')
    with open(save_folder / 'annotation.txt', 'w') as file:
        for el in result:
            x_left, x_right, y_left, y_right,  = el['x_bbox_indexes'][0], el['x_bbox_indexes'][1], el['y_bbox_indexes'][0], el['y_bbox_indexes'][1]
            # width = x_right  - x_left
            # height = y_right - y_left
            string = f'{x_left} {y_left} {x_right} {y_right} 0'
            #print(f'ANNOTATION: x_left = {x_left}, y_left = {y_left}, x_right = {x_right}, y_right = {y_right}')
            file.write(f'{string}\n')