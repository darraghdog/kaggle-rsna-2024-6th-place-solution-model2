'''
git clone https://github.com/rwindsor1/SpineNet
cd SpineNet/
pip install -r requirements.txt
mkdir -p spinenet/weights/detect-vfr

'''
import sys
import platform
import warnings
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pydicom
import sys
import glob
import torch
import os
import matplotlib.pyplot as plt
from bounding_box import bounding_box as bbfn
import pickle
from tqdm import tqdm
import gc
import warnings
import pathlib
import json
import cv2
from PIL import Image
import io
import collections
import copy
# visualize vertebrae detections in slices
from matplotlib.patches import Polygon
import matplotlib.patches as patches
gc.collect()

if platform.system() == 'Darwin':
    os.chdir('/Users/dhanley/Documents/kaggle-rsna-2024-6th-place-solution-model2')
warnings.simplefilter("ignore")
#os.environ["CUDA_VISIBLE_DEVICES"] = '7'

sys.path.append('SpineNet')
from utils import set_pandas_display, dumpobj, loadobj
import spinenet
from spinenet import SpineNet, download_example_scan
from spinenet.io import load_dicoms_from_folder


def set_pandas_display():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows',10000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
set_pandas_display()


if platform.system() == 'Darwin':
    DATA_FOLDER = "datamount"
else:
    DATA_FOLDER = "/mount/data"
TARGET_FOLDER0 = f'{DATA_FOLDER}/spinenet/vert_dicts_v0/'
TARGET_FOLDER1 = f'{DATA_FOLDER}/spinenet/ivds_v0/'
pathlib.Path(TARGET_FOLDER0).mkdir(parents=True, exist_ok=True)
pathlib.Path(TARGET_FOLDER1).mkdir(parents=True, exist_ok=True)
VERT_KEYS = 'L1 L2 L3 L4 L5 S1 T6 T7 T8 T9 T10 T11 T12 C2 C3 C4 C5 C6 C7 T1 T2 T3 T4 T5'.split()

COMP_VERTS = 'L1 L2 L3 L4 L5 S1'.split()
LEVELS = 'l1_l2 l2_l3 l3_l4 l4_l5 l5_s1'.split()

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# load in spinenet. Replace device with 'cpu' if you are not using a CUDA-enabled machine.
spinenet.download_weights(verbose=True)
spnt = SpineNet(device=DEVICE , verbose=True)


df = pd.read_csv(f'{DATA_FOLDER}/train_folded_v1.csv')
#df = df[df['series_description'] == 'Sagittal T1']
df = df[df['series_description'].str.contains("Sagittal T")]
df


trncdf = pd.read_csv('datamount/train_label_coordinates.csv')

def poly2bb(polygon, to_int = True):
    x_coords = [point[0] for point in polygon]
    y_coords =  [point[1] for point in polygon]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    if to_int:
        x_min, x_max = int(np.floor(x_min)), int(np.ceil(x_max))
        y_min, y_max = int(np.floor(y_min)), int(np.ceil(y_max))
    return [x_min, y_min, x_max, y_max]

if False:
    chkls = []
    for study_id, series_id in tqdm(df[['study_id','series_id']].values, total = len(df)):
        dicom_folder = f'{DATA_FOLDER}/train_images/{study_id}/{series_id}/'
        is_on_disk = os.path.isfile(TARGET_FOLDER0 + f'{series_id}_vert_dicts.p')
        chkls.append(is_on_disk)

        scan = load_dicoms_from_folder(dicom_folder, require_extensions=False)
        vert_dicts = spnt.detect_vb(scan.volume, scan.pixel_spacing)
        ivds = spnt.get_ivds_from_vert_dicts(scan_volume=scan.volume,vert_dicts=vert_dicts)
        with open(TARGET_FOLDER0 + f'{series_id}_vert_dicts.p','wb') as f:
            pickle.dump(vert_dicts,f)
        with open(TARGET_FOLDER1 + f'{series_id}_ivds.p','wb') as f:
            pickle.dump(ivds,f)





