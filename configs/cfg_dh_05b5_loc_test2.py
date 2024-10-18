import os
import sys
from importlib import import_module
import platform
import json
import numpy as np
import glob
from types import SimpleNamespace
import albumentations as A
import pandas as pd

if not platform.node().isdigit():
    if platform.system()=='Darwin':
        PATH = '/Users/dhanley/Documents/kaggle-rsna-2024-6th-place-solution-model2'
    else:
        PATH = './'
    os.chdir(f'{PATH}')

    sys.path.append("configs")
    sys.path.append("augs")
    sys.path.append("models")
    sys.path.append("data")
    sys.path.append("postprocess")

    from default_config import basic_cfg
    from default_cv_config import basic_cv_cfg
    import pandas as pd
    cfg = basic_cfg
    cfg.debug = True
    if platform.system()!='Darwin':
        cfg.name = os.path.basename(__file__).split(".")[0]
        cfg.output_dir = f"{PATH}/weights/{os.path.basename(__file__).split('.')[0]}"
    cfg.data_dir = f"{PATH}/datamount/"
else:
    from default_cv_config import basic_cv_cfg
    from default_config import basic_cfg
    import pandas as pd
    cfg = basic_cfg
    cfg.debug = True
    PATH = "/mount/data"
    cfg.name = os.path.basename(__file__).split(".")[0]
    cfg.output_dir = f"{PATH}/models/{os.path.basename(__file__).split('.')[0]}"
    cfg.data_dir = f"{PATH}/"


cfg = basic_cv_cfg

# paths
#cfg.name = os.path.basename(__file__).split(".")[0]
#cfg.data_dir = ""
cfg.data_folder = f'{cfg.data_dir}/'
cfg.dicom_folder = f'{cfg.data_dir}/train_images'
cfg.image_folder = f'{cfg.data_dir}/train_images_v01'
cfg.train_df = f'{cfg.data_dir}/train_folded_v1.csv'
#cfg.loc_df = f'{cfg.data_dir}/spinenet/vert_box_v01.csv.gz'
cfg.coord_df = f'{cfg.data_dir}/train_label_coordinates.csv'
#cfg.condition_filter = ['Spinal Canal Stenosis',
#                        'Right Neural Foraminal Narrowing',
#                        'Left Neural Foraminal Narrowing']
# cfg.condition_filter = ['Sagittal T2/STIR', 'Sagittal T1']
cfg.condition_filter = ['Axial T2']
cfg.verts = 'L1 L2 L3 L4 L5 S1'.split()


# stages
cfg.test = False
cfg.test_data_folder = cfg.data_dir + "test/"
cfg.train = False
cfg.train_val = False
cfg.val = True
cfg.full_test_set = True

#logging
cfg.neptune_project = "watercooled/rsna24"
cfg.neptune_connection_mode = "async"

#model
cfg.model = "mdl_dh_4a2" # "mdl_dh_4"
# cfg.backbone = 'efficientnet_b1'
cfg.backbone = 'efficientnetv2_rw_t'


cfg.image_normalization =  'simple'
cfg.pretrained = True
cfg.in_channels = 1

cfg.pool = 'avg'
cfg.gem_p_trainable = False

# DATASET
cfg.dataset = "ds_dh_4a3" # "ds_ch_4"


cfg.classes = ['x_left', 'y_left', 'x_right', 'y_right']
cfg.n_classes = len(cfg.classes) # len(cfg.target_columns) * len(cfg.classes)


# OPTIMIZATION & SCHEDULE

cfg.fold = 0
cfg.epochs = 1

cfg.lr = 2e-4
cfg.optimizer = "AdamW"
cfg.weight_decay = 1e-2
cfg.warmup = 0.1 * cfg.epochs
cfg.batch_size = 16
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 1.
cfg.num_workers = 8

#EVAL
cfg.post_process_pipeline = 'pp_dummy'
cfg.metric = 'default_metric'
# augs & tta

# Postprocess

#viz
cfg.viz = 'default_viz'
cfg.viz_first_batches = 0

#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True

cfg.img_size  = 384
cfg.crop_size = cfg.img_size
cfg.crop_buffer = 0.1
# cfg.train_aug = None
# cfg.val_aug = None
cfg.train_aug = A.Compose([
    A.Resize(cfg.img_size,cfg.img_size),
#     A.D4(p=1),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=25, p=0.7),
#     #A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.75),
#     #A.InvertImg(p=0.5),
    A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(int(0.375*512), int(0.375*512)), hole_width_range=(int(0.375*512), int(0.375*512)), p=0.8),
#     A.RandomCrop(height=96, width=192, p=1.0),
#    A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.5)
], keypoint_params=A.KeypointParams(format='xy'))


cfg.val_aug = A.Compose([
    A.Resize(cfg.img_size,cfg.img_size),
# #     #A.PadIfNeeded (min_height=256, min_width=940),
# #     # A.LongestMaxSize(cfg.image_width_orig,p=1),
# #     # A.PadIfNeeded(cfg.image_width_orig, cfg.image_height_orig, border_mode=cv2.BORDER_CONSTANT,p=1),
#     A.CenterCrop(p=1.0, height=crop_size, width=crop_size),
#     #A.Resize(cfg.img_size[0],cfg.img_size[1])
], keypoint_params=A.KeypointParams(format='xy'))

cfg.pretrained_weights = f"/mount/data/models/cfg_dh_05b5_loc/fold0/checkpoint_last_seed122775.pth"
cfg.pretrained_weights_strict = True