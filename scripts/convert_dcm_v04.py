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
from multiprocessing import Pool
import multiprocessing


if platform.system() == 'Darwin':
    os.chdir('/Users/dhanley/Documents/kaggle-rsna-2024-6th-place-solution-model2')
warnings.simplefilter("ignore")

def apply_windowing(dcm, norm_val = 255):
    pixel_data = dcm.pixel_array
    pixel_data_windowed2 = pixel_data.copy()
    min_int = np.quantile(pixel_data_windowed2, 0.005)
    max_int = np.quantile(pixel_data_windowed2, 0.995)
    pixel_data_windowed2[pixel_data < min_int] = min_int
    pixel_data_windowed2[pixel_data > max_int] = max_int
    pixel_data_windowed2 = (pixel_data_windowed2/(max_int/255)).round().astype(np.uint8)
    Image.fromarray(pixel_data_windowed2)
    return pixel_data_windowed2

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
    jpeg_dir = dcm_dir.replace('/train_images/', '/train_images_v04/')
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
