import os
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2


def cdir(func: callable):
    def wrapper(*args, **kwargs):
        cur_dir = Path(os.getcwd())
        os.chdir(Executor.main_path)
        func(*args, **kwargs)
        os.chdir(cur_dir)
    return wrapper


class Executor:
    main_path = Path('/home/max/projects/dataset_v2')
    SIMULATION = 'simulation'
    PROCESSING = 'processing'
    save_path = None


    @classmethod
    @cdir
    def generate_hologram(cls, save_folder: Path = None):
        os.chdir(cls.SIMULATION)
        os.system(f'./SARSim2')
        os.chdir(f'../{cls.PROCESSING}')
        os.system(
            f'python extract_gologram.py'
        )
        os.system(
            f'python process_gologram.py'
        )
    

    @classmethod
    @cdir
    def range_compress(cls):
        os.chdir(cls.PROCESSING)
        os.system(
            f'python range_compression.py'
        )


    @classmethod
    @cdir
    def replce_to_folder(cls, folder: Path):
        os.chdir(cls.PROCESSING)
        folder.mkdir(exist_ok=True, parents=True)
        if 'image.npy' in os.listdir():
            shutil.move('image.npy', folder / 'image.npy')
        elif 'gologram.npy' in os.listdir():
            shutil.move('gologram.npy', folder / 'gologram.npy')
        else:
            print('ERROR!')

    @classmethod
    @cdir
    def set_save_dir(cls, folder: Path):
        os.chdir(cls.PROCESSING)
        folder.mkdir(exist_ok=True, parents=True)
        cls.save_path = folder


    @classmethod
    def _move_file(cls, file_name: str):
        if cls.save_path is not None:
            shutil.move(file_name, cls.save_path / file_name)


    @classmethod
    @cdir
    def draw_hologram(cls):
        os.chdir(cls.PROCESSING)
        os.system(
            f'python draw_gologram.py'
        )

    @classmethod
    @cdir
    def draw_image(cls):
        os.chdir(cls.PROCESSING)
        os.system(
            f'python draw_image.py'
        )   


    @classmethod
    @cdir
    def draw_range_compressed(cls):
        os.chdir(cls.PROCESSING / cls.save_path)
        img = np.abs(np.load('gologram_range_compressed.npy'))
        plt.imshow(img)
        plt.show()


    @classmethod
    @cdir
    def show_image(cls):
        os.chdir(cls.PROCESSING / cls.save_path)
        img = np.abs(np.load('image.npy'))
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = cv2.rectangle(img, pt1=(1760, 1237), pt2=(1760 + 800, 1237 - 800), color=(255, 0, 0), thickness=2)
        img = cv2.circle(img, (1760, 1237), 10, color=(128))
        plt.imshow(img, cmap='gray')
        plt.show()


    @classmethod
    @cdir
    def show_gologram(cls):
        os.chdir(cls.PROCESSING / cls.save_path)
        #os.system('ls -l')
        hologram = np.load('gologram.npy')
        img = np.zeros((hologram.shape[0], hologram.shape[1], 3))
        img[:, :, 0] = np.real(hologram)
        img[:, :, 1] = np.imag(hologram)
        img[:, :, 2] = np.abs(hologram)
        img = 1 * (img - img.min()) / (img.max() - img.min())
        p1x = hologram.shape[1] // 2
        p1y = hologram.shape[0] // 2 - 100
        print(f'px = {p1x}, py = {p1y}')
        img = cv2.rectangle(img, pt1=(p1x - 400, p1y - 400), pt2=(p1x + 400, p1y + 400), color=(1, 1, 1), thickness=2)
        
        img = cv2.rectangle(img, pt1=(p1x - 400, p1y - 400), pt2=(p1x + 400, p1y + 400), color=(1, 1, 1), thickness=2)
        

        plt.imshow(img, cmap='gray')
        plt.show()