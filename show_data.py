import os
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path



def get_pixel_distribution(img_path: Path):
    # h, w
    img = np.abs(np.load(img_path / 'image.npy'))
    img = (255 * (img - img.min()) / (img.max() - img.min()))#.astype(np.uint8)
    center_point = (1769, 1237)
    shapes = (800, -800)
    new_img = img
    # new_img: np.ndarray = img[1237 - 400:1237 + 400, 1760 - 400:1760 + 400]
    new_img = 5 * new_img
    new_img[new_img > 255] = 255
    new_img[new_img < 2] = 0
    
    print(f'new_shape = {new_img.shape}')
    plt.imshow(new_img, cmap='gray')
    plt.show()
    new_img = new_img[new_img > 0]
    distr = new_img.flatten()
    #plt.hist(distr, bins=1000)
    #plt.show()


def show_sar(folder: Path):
    img = np.abs(np.load(folder / 'image.npy'))
    plt.title('SAR', fontsize=18)
    plt.imshow(img, cmap='gray')
    plt.show()


def show_hologram(folder: Path):
    holo = np.load(folder / 'gologram.npy')

    plt.title('Re hologram', fontsize=18)
    plt.imshow(np.real(holo))
    plt.show()

    # plt.title('Abs hologram', fontsize=18)
    # plt.imshow(np.abs(holo), cmap='gray')
    # plt.show()

    holo_compressed = np.load(folder / 'gologram_range_compressed.npy')
    plt.title('Abs hologram', fontsize=18)
    plt.imshow(np.abs(holo_compressed), cmap='gray')
    plt.show()

if __name__ == "__main__":
    MAIN_PATH = Path('/home/max/projects/dataset_v2')
    default_ship = MAIN_PATH / Path('processing/ships_default')
    standart_path = MAIN_PATH / Path('processing')
    #show_sar(default_ship)
    ship_x_50 = MAIN_PATH / Path('processing/ship_scaled_x50')
    ship_one = MAIN_PATH / Path('processing/ship_one')
    # show_hologram(Path('processing'))
    ship_1 = MAIN_PATH / Path('processing/P0002_5260_6060_6000_6800')
    #show_sar(ship_1)
    ships_data = MAIN_PATH / Path('processing/ships_data')
    #
    file_name = os.listdir(ships_data)[0]
    print(f'file_name = {file_name}')
    show_sar(ships_data / file_name)    
    get_pixel_distribution(ships_data / file_name)
    show_hologram(ships_data / file_name)