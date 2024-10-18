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
import cv2

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
#cfg.coord_df = f'{cfg.data_dir}/train_label_coordinates_theo_2808.csv'
cfg.coord_df = f'{cfg.data_dir}/sag_xy/shifted_spinenet_v02.csv.gz'
#cfg.condition_filter = ['Spinal Canal Stenosis',
#                        'Right Neural Foraminal Narrowing',
#                        'Left Neural Foraminal Narrowing']
cfg.condition_filter = ['Sagittal T2/STIR', 'Sagittal T1']
cfg.err_series_id = '47218878 123984223 268383691 372716479 406090885 463429448 1022282707 1036019370 1044930265 1046792123 1082225960 1199403131 1212326388 1487416663 1513597136 1598245081 1724042057 1814640980 1829533928 2064652967 2105425746 2154355601 2256490430 2405003641 2594441574 2711787415 2760514784 2805172442 2945488620 2991123343 3008943156 3075018987 3096848587 3237930365 3266118920 3526728886 3573989834 3582097958 3620276777 3625340069 3647714587 3657125167 3791568557 3805861667 3893961572 3913220962 3960750582 3978433476 4075717149 4076103046 4116372721'


# cfg.condition_filter = ['Axial T2']
cfg.verts = 'L1 L2 L3 L4 L5 S1'.split()
cfg.levels = 'l1_l2 l2_l3 l3_l4 l4_l5 l5_s1'.split()


# stages
cfg.test = False
cfg.test_data_folder = cfg.data_dir + "test/"
cfg.train = True
cfg.train_val = False
cfg.full_test_set = False

#logging
cfg.neptune_project = "watercooled/rsna24"
cfg.neptune_connection_mode = "async"

#model
cfg.model = "mdl_dh_4b2" # "mdl_dh_4"
# cfg.backbone = 'efficientnet_b1'
cfg.backbone = 'efficientnetv2_rw_s'


cfg.image_normalization =  'simple'
cfg.pretrained = True
cfg.in_channels = 5
assert cfg.in_channels % 2 == 1

cfg.pool = 'avg'
cfg.gem_p_trainable = False

# DATASET
cfg.dataset = "ds_dh_11g" # "ds_ch_4"

cfg.classes = ['x__l1_l2',
                 'y__l1_l2',
                 'x__l2_l3',
                 'y__l2_l3',
                 'x__l3_l4',
                 'y__l3_l4',
                 'x__l4_l5',
                 'y__l4_l5',
                 'x__l5_s1',
                 'y__l5_s1']
cfg.n_classes = len(cfg.classes) # len(cfg.target_columns) * len(cfg.classes)


# OPTIMIZATION & SCHEDULE

cfg.fold = 0
cfg.epochs = 8

cfg.lr = 2e-4
cfg.optimizer = "AdamW"
cfg.weight_decay = 1e-2
cfg.warmup = 0.1 * cfg.epochs
cfg.batch_size = 32
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 1.
cfg.num_workers = 8
cfg.mask_loss = 0.02

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

cfg.img_size  = 512
cfg.crop_size = cfg.img_size
cfg.crop_buffer = 0.1
# cfg.train_aug = None
# cfg.val_aug = None
cfg.train_aug = A.Compose([
    A.Resize(cfg.img_size,cfg.img_size, interpolation=cv2.INTER_CUBIC),
#     A.D4(p=1),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.7),
#     #A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.75),
#     #A.InvertImg(p=0.5),
#    A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.5)
    A.CoarseDropout(num_holes_range=(10, 30),
                    hole_height_range=(int(0.01*cfg.img_size), int(0.08*cfg.img_size)),
                    hole_width_range=(int(0.01*cfg.img_size), int(0.08*cfg.img_size)), p=0.8),
    A.RandomCrop(height=cfg.img_size, width=3 * cfg.img_size // 4, p=1.0),

], keypoint_params=A.KeypointParams(format='xy'))


cfg.val_aug = A.Compose([
    A.Resize(cfg.img_size,cfg.img_size, interpolation=cv2.INTER_CUBIC),
# #     #A.PadIfNeeded (min_height=256, min_width=940),
# #     # A.LongestMaxSize(cfg.image_width_orig,p=1),
# #     # A.PadIfNeeded(cfg.image_width_orig, cfg.image_height_orig, border_mode=cv2.BORDER_CONSTANT,p=1),
#     A.CenterCrop(p=1.0, height=crop_size, width=crop_size),
#     #A.Resize(cfg.img_size[0],cfg.img_size[1])
], keypoint_params=A.KeypointParams(format='xy'))

