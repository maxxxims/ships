from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path



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

    plt.title('Abs hologram', fontsize=18)
    plt.imshow(np.abs(holo), cmap='gray')
    plt.show()

if __name__ == "__main__":
    MAIN_PATH = Path('/home/max/projects/dataset_v2')
    default_ship = MAIN_PATH / Path('processing/ships_default')
    standart_path = MAIN_PATH / Path('processing')
    #show_sar(default_ship)
    ship_x_50 = MAIN_PATH / Path('processing/ship_scaled_x50')
    ship_one = MAIN_PATH / Path('processing/ship_one')
    # show_hologram(Path('processing'))
    show_sar(standart_path)