import os
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Union


def cdir(func: callable):
    def wrapper(*args, **kwargs):
        cur_dir = Path(os.getcwd())
        os.chdir(Executor.main_path)
        func(*args, **kwargs)
        os.chdir(cur_dir)
    return wrapper


def normalize_arr(data: np.ndarray, scale_factor: int = 255, as_uint8: bool = True) -> np.ndarray:
    data = scale_factor * (data - data.min()) / (data.max() - data.min())
    if as_uint8:
        data = data.astype(np.uint8)
    return data

class Executor:
    main_path = Path('/home/max/projects/dataset_v2')
    SIMULATION = 'simulation'
    PROCESSING = 'processing'
    save_path = None


    @classmethod
    @cdir
    def generate_output_signal(cls):
        os.chdir(cls.SIMULATION)
        os.system(f'./SARSim2')

    @classmethod
    @cdir
    def generate_hologram(cls, save_folder: Path = None):
        
        
        os.chdir(f'{cls.PROCESSING}')
        os.system(
            f'python extract_gologram.py'
        )
        os.system(
            f'python process_gologram.py'
        )
        #os.chdir(f'../{cls.PROCESSING}')
        
    

    @classmethod
    @cdir
    def range_compress(cls):
        os.chdir(cls.PROCESSING)
        os.system(
            f'python range_compression.py'
        )

    @classmethod
    @cdir
    def move_result_file_to_save_folder(cls):
        os.chdir(cls.PROCESSING)
        for file_name in ['gologram.npy', 'image.npy', 'gologram_range_compressed.npy']:
            if file_name in os.listdir():
                cls._move_file(file_name)
            else:
                print(f'{file_name} not in processing folder')
                os.system('ls -l')


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
    def set_save_dir(cls, folder: Union[Path, str] = None):
        if folder is None:
            cls.save_path = None
            return
        if isinstance(folder, str):
            folder = Path(folder)
        os.chdir(cls.PROCESSING)
        folder.mkdir(exist_ok=True, parents=True)
        cls.save_path = folder



    @classmethod
    def _open_save_folder(cls):
        os.chdir(cls.PROCESSING)
        if cls.save_path is not None:
            os.chdir(cls.save_path)


    @classmethod
    def _move_file(cls, file_name: str):
        #os.chdir(cls.PROCESSING)
        if cls.save_path is not None:
            shutil.move(file_name, cls.save_path / file_name)
        else:
            print(f'{file_name} cannot be moved')


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
        #os.chdir(cls.PROCESSING / cls.save_path)
        cls._open_save_folder()
        # os.system('ls -l')
        img = np.abs(np.load('gologram_range_compressed.npy'))
        plt.imshow(img)
        plt.show()


    @classmethod
    @cdir
    def show_image(cls, cutted=False, show_annotation: bool = True):
        #os.chdir(cls.PROCESSING / cls.save_path)
        cls._open_save_folder()
        """img = np.abs(np.load('image.npy'))
        img = 255 * (img - img.min()) / (img.max() - img.min())
        if not cutted:
            img = cv2.rectangle(img, pt1=(1760 - 400, 1237 - 400), pt2=(1760 + 400, 1237 + 400), color=(255, 0, 0), thickness=2)
            img = cv2.circle(img, (1760, 1237), 10, color=(128))"""
            
            #img = img[1237 - 400:1237 + 400, 1760 - 400:1760 + 400]
        img = plt.imread('image.png')
        if show_annotation:
            def convert_coord(x: float, y: float):
                x_new = x - (1760 - 640)
                y_new = y - (1237 - 640)
                return (int(x_new), int(y_new))
                #return (int(x_new), int(y_new))
            

            annotation = np.loadtxt('annotation.txt', ndmin=2)
            for a in annotation:
                #a = a.astype(int)
                img = cv2.rectangle(img, pt1=convert_coord(a[0], a[1]), pt2=convert_coord(a[2], a[3]), color=(1, 0, 0, 1), thickness=2)
                print(f'{convert_coord(a[0], a[1])}; {convert_coord(a[2], a[3])}; was = {a[0], a[1]}')
        plt.imsave('image_cutted.png', img)
        #plt.imshow(img, cmap='gray')
        #plt.show()


    @classmethod
    @cdir
    def show_gologram(cls, cutted=False):
        #os.chdir(cls.PROCESSING / cls.save_path)
        #os.system('ls -l')
        cls._open_save_folder()
        hologram = np.load('gologram.npy')
        img = np.zeros((hologram.shape[0], hologram.shape[1], 3))
        img[:, :, 0] = np.real(hologram)
        img[:, :, 1] = np.imag(hologram)
        img[:, :, 2] = np.abs(hologram)
        img = 1 * (img - img.min()) / (img.max() - img.min())
        img = np.real(hologram)
        if not cutted:
            p1x = hologram.shape[1] // 2
            p1y = hologram.shape[0] // 2 #- 100
            img = cv2.rectangle(img, pt1=(p1x - 400, p1y - 400), pt2=(p1x + 400, p1y + 400), color=(1, 1, 1), thickness=2)
            
            img = cv2.rectangle(img, pt1=(p1x - 400, p1y - 400), pt2=(p1x + 400, p1y + 400), color=(1, 1, 1), thickness=2)
            

        plt.imshow(img, cmap='gray')
        plt.show()



    @classmethod
    @cdir
    def cut_image(cls, save_png: bool = False):
        cls._open_save_folder()
        img = np.abs(np.load('image.npy'))
        img = img[1237 - 640: 1237 + 640, 1760 - 640 : 1760 + 640]
        if not save_png:    np.save('image.npy', img)
        else:   
            img = normalize_arr(img, as_uint8=True)
            plt.imsave('image.png', img, cmap='gray')
            os.remove('image.npy')
            
    @classmethod
    @cdir
    def cut_hologram(cls, save_png: bool = False):
        cls._open_save_folder()
        hologram = np.load('gologram.npy')
        # print(f'hologram shape = {hologram.shape}')
        # plt.imshow(np.real(hologram))
        # plt.show()

        h, w = hologram.shape[0] // 2, hologram.shape[1] // 2
        hologram = hologram[h - 640: h + 640, w - 640 : w + 640]
        # plt.imshow(np.real(hologram))
        # plt.show()
        if not save_png:    np.save('gologram.npy', hologram)
        else:   
            hologram_img = np.zeros((hologram.shape[0], hologram.shape[1], 3))
            hologram_img[:, :, 0] = normalize_arr(np.real(hologram))
            hologram_img[:, :, 1] = normalize_arr(np.real(hologram))
            hologram_img[:, :, 2] = normalize_arr(np.sqrt(np.real(hologram)**2 +  np.sqrt(np.imag(hologram)**2)))
            plt.imsave('gologram.png', hologram_img.astype(np.uint8), cmap='gray')
            os.remove('gologram.npy')


    @classmethod
    @cdir
    def cut_comressed_hologram(cls, save_png: bool = False):
        cls._open_save_folder()
        hologram = np.load('gologram_range_compressed.npy')
        #print(f'hologram shape = {hologram.shape}')
        #plt.imshow(np.real(hologram))
        #plt.show()

        #h, w = hologram.shape[0] // 2, hologram.shape[1] // 2
        hologram = hologram[1237 - 640: 1237 + 640, 1760 - 640 : 1760 + 640]
        #plt.imshow(np.real(hologram))
        #plt.show()
        if not save_png:    np.save('gologram_range_compressed.npy', hologram)
        else:   
            hologram_img = np.zeros((hologram.shape[0], hologram.shape[1], 3))
            hologram_img[:, :, 0] = normalize_arr(np.real(hologram))
            hologram_img[:, :, 1] = normalize_arr(np.real(hologram))
            hologram_img[:, :, 2] = normalize_arr(np.sqrt(np.real(hologram)**2 +  np.sqrt(np.imag(hologram)**2)))
            plt.imsave('gologram_range_compressed.png', hologram_img.astype(np.uint8), cmap='gray')
            os.remove('gologram_range_compressed.npy')