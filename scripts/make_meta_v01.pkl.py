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
import torch
import pylibjpeg
import pydicom
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from multiprocessing import Pool
from bounding_box import bounding_box as bbfn
import json
import sys
import multiprocessing as mp


if platform.system() == 'Darwin':
    os.chdir('/Users/dhanley/Documents/kaggle-rsna-2024-6th-place-solution-model2')
warnings.simplefilter("ignore")

def get_img_names(SEARCH_PATH):
    imgnms = glob.glob(SEARCH_PATH)
    imgnms_ids = [int(i.split('/')[-1].split('.')[0]) for i in imgnms]
    imgnms = [j for i,j in sorted(zip(imgnms_ids, imgnms))]
    imgnms_ids = [int(i.split('/')[-1].split('.')[0]) for i in imgnms]
    return imgnms, imgnms_ids

def set_pandas_display():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows',10000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
set_pandas_display()

'''
PixelSpacing SpacingBetweenSlices SliceThickness SliceLocation
'''

def process_file(fnm):
    d = pydicom.dcmread(fnm)
    d1 = dict(ImagePositionPatient=tuple([float(v) for v in d.ImagePositionPatient]),
              ImageOrientationPatient=tuple([float(v) for v in d.ImageOrientationPatient]),
              PixelSpacing=tuple([float(v) for v in d.PixelSpacing]),
              SpacingBetweenSlices=float(d.SpacingBetweenSlices),
              SliceThickness=float(d.SliceThickness),
              SliceLocation=float(d.SliceLocation),
              PatientID = d.PatientID,
              PatientPosition = d.PatientPosition,
              SeriesDescription=d.SeriesDescription,
              PhotometricInterpretation=d.PhotometricInterpretation,
              img_w = int(d.Columns),
              img_h = int(d.Rows))
    d1['study_id'], d1['series_id'], d1['instance_number'] = map(int, os.path.splitext(fnm)[0].split('/')[-3:])
    return d1

if __name__ == '__main__':
    fnmls = sorted(glob.glob("datamount/train_images/*/*/*"))
    # metals = [process_file(fnm) for fnm in tqdm(fnmls)]

    with mp.Pool(mp.cpu_count()) as pool:
        metals = list(tqdm(pool.imap(process_file, fnmls), total=len(fnmls)))

    metadf = pd.DataFrame(metals).sort_values('series_id instance_number'.split())
    metadf['instance_number'] = metadf.instance_number.astype(int)
    del  metals
    gc.collect()
    metadf.to_pickle("datamount/meta_v1.pkl")
    print(metadf.dtypes)
    metadf.head(10)

