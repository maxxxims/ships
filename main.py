from pathlib import Path
from draw_ships import open_json_file, get_image_annotation, make_annotation_for_all_ships, save_result
from make_coords import convert_to_coords, save_coords
import numpy as np
from matplotlib import pyplot as plt



if __name__ == "__main__":
    path_to_labels = Path('draw/labels.json')
    path_to_image_folder = Path('draw')
    save_temporary_result_path = Path('draw/result.json')
    save_path = Path('/home/max/projects/dataset_v2/simulation/group_0_data.json')  # path to json file with coords for generator


    annotation_data = open_json_file(path_to_labels)   # dict with special data for image annotation in dataset

    image, annotations = get_image_annotation(annotation_data, file_name='P0001_1200_2000_8400_9200.jpg')
    img = np.copy(plt.imread(path_to_image_folder / image['file_name']))
    result = make_annotation_for_all_ships(img, annotations, new_points=20, scale_img=True)
    save_result(image['file_name'], result, save_result_path=save_temporary_result_path)


    image_index_data = open_json_file(save_temporary_result_path)
    coords = convert_to_coords(image_index_data, limit = None)
    save_coords(coords, save_path=save_path)