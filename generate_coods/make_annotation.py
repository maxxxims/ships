import json
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from make_annotation import AnnotationTransformer



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


def save_annotation_v2(result: list, save_folder: Path, transformer: AnnotationTransformer):
    """
        write bbox coords in full-scaled SAR image in line and col format by Ilia's code
    """
    # with open(save_folder / 'annotation_abs.txt', 'w') as file:
    #     for el in result:
    #         assert len(el['x_bbox_indexes']) == 2 == len(el['y_bbox_indexes']) == len(el['z_bbox_indexes'])
    #         x_left, x_right, y_left, y_right, z_left, z_right = el['x_bbox_indexes'][0], el['x_bbox_indexes'][1], el['y_bbox_indexes'][0], el['y_bbox_indexes'][1], el['z_bbox_indexes'][0], el['z_bbox_indexes'][1]
            
    #         (line_left, col_left), (line_right, col_right) = transformer.transform_bbox(x_bbox_indexes=[x_left, x_right], y_bbox_indexes=[y_left, y_right], z_bbox_indexes=[z_left, z_right])
    #         file.write(f'{line_left} {col_left} {line_right} {col_right}\n')

    data = {
        'abs_bbox_coord': []
    }
    with open(save_folder / 'annotation.json', 'w') as f:
        for el in result:
            assert len(el['x_bbox_indexes']) == 2 == len(el['y_bbox_indexes']) == len(el['z_bbox_indexes'])
            x_left, x_right, y_left, y_right, z_left, z_right = el['x_bbox_indexes'][0], el['x_bbox_indexes'][1], el['y_bbox_indexes'][0], el['y_bbox_indexes'][1], el['z_bbox_indexes'][0], el['z_bbox_indexes'][1]
            (line_left, col_left), (line_right, col_right) = transformer.transform_bbox(x_bbox_indexes=[x_left, x_right], y_bbox_indexes=[y_left, y_right], z_bbox_indexes=[z_left, z_right])
            data['abs_bbox_coord'].append({
                'line_left': line_left, 'col_left': col_left, 'line_right': line_right, 'col_right': col_right
            })
        
        json.dump(data, f, indent=2)
