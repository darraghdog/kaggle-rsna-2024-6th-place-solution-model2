import pandas as pd
import numpy as np
import warnings, copy, collections
import platform
import os
import argparse
import string
import gc
import glob
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageStat
import cv2
import imagesize
import pylibjpeg
import pydicom
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from multiprocessing import Pool
import imagesize
import pylibjpeg
import pydicom
from multiprocessing import Pool
import multiprocessing

if platform.system() == 'Darwin':
    os.chdir('/Users/dhanley/Documents/kaggle-rsna-2024-6th-place-solution-model2')
warnings.simplefilter("ignore")

def apply_windowing(dcm, norm_val = 255):
    window_center = int(dcm.WindowCenter)
    window_width = int(dcm.WindowWidth)
    pixel_data = dcm.pixel_array
    min_int = window_center - window_width // 2
    max_int = window_center + window_width // 2
    pixel_data_windowed = pixel_data.copy()
    pixel_data_windowed[pixel_data < min_int] = min_int
    pixel_data_windowed[pixel_data > max_int] = max_int
    pixel_data_windowed = (pixel_data_windowed/(max_int/255)).round().astype(np.uint8)
    return pixel_data_windowed


DATADIR = 'datamount/'
fnmls = sorted(glob.glob(f"{DATADIR}/train_images/*/*/*"))
labels = "normal_mild/moderate/severe".split('/')
trndf = pd.read_csv(f'{DATADIR}/train.csv')
trndf.iloc[0]


trnmdf = pd.read_csv(f'{DATADIR}/train_label_coordinates.csv')

len(fnmls)

def worker(fnm):
    fpath = Path(fnm)
    dcm_dir = str(fpath.parent)
    jpeg_dir = dcm_dir.replace('/train_images/', '/train_images_v01/')
    Path(jpeg_dir).mkdir(parents=True, exist_ok=True)
    dcmfile = fnm
    dcm = pydicom.dcmread(dcmfile)
    img = apply_windowing(dcm)
    jpeg_nm = f"{jpeg_dir}/{fpath.stem}.jpeg"
    cv2.imwrite(jpeg_nm, img)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(4)
    for _ in tqdm(pool.imap_unordered(worker, fnmls), total=len(fnmls)):
        pass

    pool.close()
    pool.join()

