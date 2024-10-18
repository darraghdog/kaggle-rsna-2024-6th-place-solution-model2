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
sys.path.append('SpineNet')
from utils import set_pandas_display, dumpobj, loadobj
import spinenet
from spinenet import SpineNet, download_example_scan
from spinenet.io import load_dicoms_from_folder


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


if platform.system() == 'Darwin':
    DATA_FOLDER = "datamount"
else:
    DATA_FOLDER = "/mount/data"
TARGET_FOLDER0 = f'{DATA_FOLDER}/spinenet/vert_dicts_v0/'
IMAGE_FOLDER = f'{DATA_FOLDER}/train_images_v01'

CLASSES = ['x__l1_l2',
                 'y__l1_l2',
                 'x__l2_l3',
                 'y__l2_l3',
                 'x__l3_l4',
                 'y__l3_l4',
                 'x__l4_l5',
                 'y__l4_l5',
                 'x__l5_s1',
                 'y__l5_s1']

COMP_VERTS = 'T11 T12 L1 L2 L3 L4 L5 S1 S2'.split()
LEVELS = [f'{i}_{j}'.lower() for i,j in  zip(COMP_VERTS, COMP_VERTS[1:])]
COMP_VERTS_OUT = 'L1 L2 L3 L4 L5 S1'.split()
LEVELS_OUT = [f'{i}_{j}'.lower() for i,j in  zip(COMP_VERTS_OUT, COMP_VERTS_OUT[1:])]


condition_filter = ['Sagittal T1', 'Sagittal T2/STIR']
condition_map = {'Sagittal T1': "neural_foraminal_narrowing",
                     'Sagittal T2/STIR': "spinal_canal_stenosis",
                     'Axial T2': "subarticular_stenosis"}

fnmls = sorted(glob.glob("datamount/train_images/*/*/*"))
labels = "normal_mild/moderate/severe".split('/')
trndf = pd.read_csv('datamount/train.csv')
trnfdf = pd.read_csv('datamount/train_folded_v1.csv')
trnfdf = trnfdf[trnfdf.series_description.isin(condition_filter )]
series_id_filt = trnfdf.series_id.unique()


trncdf = pd.read_csv('datamount/train_label_coordinates.csv')
trncdf['level'] = trncdf['level'].str.lower().str.replace('/','_')
trncdf = trncdf[trncdf.series_id.isin(series_id_filt)]
metadf = pd.read_pickle("datamount/meta_v1.pkl")
trncdf = pd.merge(trncdf, metadf['series_id img_h  img_w'.split()].drop_duplicates(), on = 'series_id', how = 'left')
trncdf = trncdf.set_index("series_id")
trncdf['x_norm'] = trncdf['x'] / trncdf['img_w']
trncdf['y_norm'] = trncdf['y'] / trncdf['img_h']


'''
Get the point for condition
'''

ddf = trnfdf [['study_id','series_id']].drop_duplicates()
vdfls = []
for r in tqdm(ddf.itertuples(), total = len(ddf)):
    outd = collections.defaultdict(list)
    vert_dicts = loadobj(f'{TARGET_FOLDER0}/{r.series_id}_vert_dicts.p')
    vd0 = [v for v in vert_dicts if v['predicted_label'] in COMP_VERTS]
    base_img_dir = f'{IMAGE_FOLDER}/{r.study_id}/{r.series_id}/*.*'
    imgnms, imgnms_ids = get_img_names(base_img_dir)
    if len(vd0) == 0: continue
    for v in vd0:
        num_slices = len(v['slice_nos'])
        outd['instance_idx'] += v['slice_nos']
        outd['instance_number'] += [imgnms_ids[i] for i in v['slice_nos']]
        outd['series_id'] += [r.series_id] * num_slices
        outd['study_id'] += [r.study_id] * num_slices
        outd['vertebrae'] += [v['predicted_label']] * num_slices
        outd['polygon'] += [json.dumps([[round(j, 2) for j in i] for i in p]) for p in v['polys']]
    vdfls.append(pd.DataFrame(outd))
vdf = pd.concat(vdfls).reset_index(drop = True)
vdf.groupby('series_id')['vertebrae'].nunique().value_counts()


vdf = vdf.sort_values('series_id study_id instance_number vertebrae'.split()).reset_index(drop = True)
vdf['vertebrae_pre'] = vdf.groupby('series_id study_id instance_number'.split())['vertebrae'].transform('shift').fillna('')
vdf['polygon_pre'] = vdf.groupby('series_id study_id instance_number'.split())['polygon'].transform('shift').fillna('')
vdf['level'] = vdf['vertebrae_pre'].str.lower() + '_' + vdf['vertebrae'].str.lower()
vdf = vdf[vdf.level.isin(LEVELS)].drop(columns='vertebrae_pre vertebrae'.split())
vdf = vdf.reset_index(drop = True)
centers = []
for r in vdf.itertuples():
    polys = np.array([json.loads(getattr(r, c)) for c in 'polygon_pre polygon'.split()])
    center = np.stack((polys[0,1] * 0.7,
                       polys[1,0] * 0.3)).sum(0)
    center = center.round().astype(int).tolist()
    centers.append(center)
vdf['x y'.split()] = centers
vdf = vdf.drop(columns = "polygon polygon_pre".split())

vdf['shift'] = 0
mapper1 = {k:v for k,v in zip(LEVELS[1:], LEVELS[:])}
mapper2 = {k:v for k,v in zip(LEVELS[:], LEVELS[1:])}
vdf1 = vdf.copy()
vdf2 = vdf.copy()
vdf1['level'] = vdf1.level.map(mapper1) # '{"t12_l1": "t11_t12", "l1_l2": "t12_l1", "l2_l3": "l1_l2", "l3_l4": "l2_l3", "l4_l5": "l3_l4", "l5_s1": "l4_l5", "s1_s2": "l5_s1"}'
vdf2['level'] = vdf2.level.map(mapper2) # '{"t11_t12": "t12_l1", "t12_l1": "l1_l2", "l1_l2": "l2_l3", "l2_l3": "l3_l4", "l3_l4": "l4_l5", "l4_l5": "l5_s1", "l5_s1": "s1_s2"}'
vdf1['shift'] = 1   # '{"t12_l1": "t11_t12", "l1_l2": "t12_l1", "l2_l3": "l1_l2", "l3_l4": "l2_l3", "l4_l5": "l3_l4", "l5_s1": "l4_l5", "s1_s2": "l5_s1"}'
vdf2['shift'] = -1  # '{"t11_t12": "t12_l1", "t12_l1": "l1_l2", "l1_l2": "l2_l3", "l2_l3": "l3_l4", "l3_l4": "l4_l5", "l4_l5": "l5_s1", "l5_s1": "s1_s2"}'

vdf = pd.concat([vdf, vdf1.dropna(), vdf2.dropna()])

vdfdiff = pd.merge(vdf, \
         trncdf.reset_index()['series_id instance_number level x_norm y_norm img_h img_w'.split()],
         on = "series_id instance_number level".split(), how = 'inner')

vdfdiff['x_normd'] = vdfdiff['x'] / vdfdiff['img_w']
vdfdiff['y_normd'] = vdfdiff['y'] / vdfdiff['img_h']
vdfdiff['diff'] = ((vdfdiff['y_normd']-vdfdiff['y_norm']) ** 2 + (vdfdiff['x_normd']-vdfdiff['x_norm']) ** 2) ** 0.5
vdfdiff = vdfdiff.groupby('series_id shift'.split())['diff'].mean().reset_index().sort_values('series_id  diff'.split())
vdfdiff['sequence'] = vdfdiff.groupby('series_id').cumcount()
buffer = vdfdiff.query('sequence==1')['diff'].values - vdfdiff.query('sequence==0')['diff'].values
vdfdiff['buffer'] = buffer.repeat(3)
vdfdiff = vdfdiff.query('sequence==0')

THRESH_BUFFER = 0.03
vdfdiff = vdfdiff.query('buffer>@THRESH_BUFFER')
THRESH_DIFF = 0.06
vdfdiff = vdfdiff.query('diff < @THRESH_DIFF')


vdf2 = pd.merge(vdf, vdfdiff['series_id  shift'.split()], on = 'series_id  shift'.split(), how = 'inner')
vdf2 = vdf2.drop(columns = 'instance_idx'.split())
vdf2 = vdf2[vdf2.level.isin(LEVELS_OUT)].reset_index(drop = True)
vdf2.level.value_counts()

vdf2.to_csv('datamount/sag_xy/shifted_spinenet_v02.csv.gz', index = False)







