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
cfg.image_folder = f'{cfg.data_dir}/train_images_v04'
cfg.train_df = f'{cfg.data_dir}/train_folded_v1.csv'
cfg.meta_df = f'{cfg.data_dir}/meta_v02.pkl'

#cfg.loc_df = f'{cfg.data_dir}/spinenet/vert_box_v01.csv.gz'
#cfg.loc_df = f'{cfg.data_dir}/sag_xy/spinenet_xmax_ybetween.csv.gz'
#cfg.loc_df = f'{cfg.data_dir}/sag_xy/shifted_spinenet_v01.csv.gz'
cfg.loc_df = f'{cfg.data_dir}/sag_xy/test__cfg_dh_14p2_locsag_test.csv.gz'
#cfg.side_df = f'{cfg.data_dir}/sag_t1_side/pred__cfg_dh_10c.csv'
cfg.coord_df = f'{cfg.data_dir}/train_label_coordinates.csv'

# cfg.mapax_df = f'{cfg.data_dir}/axial_lvl/test__cfg_dh_14p2_locsag_mapped_v01.csv.gz'
cfg.lvl_df = f'{cfg.data_dir}/axial_lvl/cfg_dh_07o_locax.csv.gz'
cfg.locax_df = f'{cfg.data_dir}/axial_xy/cfg_dh_05b5_loc_test2.csv.gz'
cfg.mapax_df = f'{cfg.data_dir}/axial_lvl/test__dh_14p2___dh_14s10a____locsag_mapped_v02.pkl'
#cfg.condition_filter = ['Spinal Canal Stenosis',
#                        'Right Neural Foraminal Narrowing',
#                        'Left Neural Foraminal Narrowing']
cfg.series_descriptions = ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']
cfg.condition_filter = [ 'Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']
cfg.condition_map = {'Sagittal T1': "neural_foraminal_narrowing",
                     'Sagittal T2/STIR': "spinal_canal_stenosis",
                     'Axial T2': "subarticular_stenosis"}
cfg.parts = ['ss'] #['ss', 'nfn', 'scs']
cfg.targets = [
        'spinal_canal_stenosis_l1_l2',
        'spinal_canal_stenosis_l2_l3',
        'spinal_canal_stenosis_l3_l4',
        'spinal_canal_stenosis_l4_l5',
        'spinal_canal_stenosis_l5_s1',
         'left_neural_foraminal_narrowing_l1_l2',
         'left_neural_foraminal_narrowing_l2_l3',
         'left_neural_foraminal_narrowing_l3_l4',
         'left_neural_foraminal_narrowing_l4_l5',
         'left_neural_foraminal_narrowing_l5_s1',
         'right_neural_foraminal_narrowing_l1_l2',
         'right_neural_foraminal_narrowing_l2_l3',
         'right_neural_foraminal_narrowing_l3_l4',
         'right_neural_foraminal_narrowing_l4_l5',
         'right_neural_foraminal_narrowing_l5_s1',
         'left_subarticular_stenosis_l1_l2',
         'left_subarticular_stenosis_l2_l3',
         'left_subarticular_stenosis_l3_l4',
         'left_subarticular_stenosis_l4_l5',
         'left_subarticular_stenosis_l5_s1',
         'right_subarticular_stenosis_l1_l2',
         'right_subarticular_stenosis_l2_l3',
         'right_subarticular_stenosis_l3_l4',
         'right_subarticular_stenosis_l4_l5',
         'right_subarticular_stenosis_l5_s1']


cfg.verts = 'L1 L2 L3 L4 L5 S1'.split()
cfg.levels = 'l1_l2 l2_l3 l3_l4 l4_l5 l5_s1'.split()
cfg.slicelocation_clip = 80
cfg.add_series_type_emb = False
cfg.add_level_emb = True

# stages
cfg.test = False
cfg.test_data_folder = cfg.data_dir + "test/"
cfg.train = True
cfg.train_val = False

#logging
cfg.neptune_project = "watercooled/rsna24"
cfg.neptune_connection_mode = "async"

#model
cfg.model = "mdl_dh_23a" # "mdl_dh_4"
cfg.backbone = 'efficientnetv2_rw_t'

# Sequence model
cfg.hidden_dim = 256 * 3

cfg.image_normalization =  'simple'
cfg.pretrained = True
cfg.in_channels = 1

cfg.pool = 'avg'
cfg.gem_p_trainable = False

# DATASET
cfg.dataset = "ds_dh_23a" # "ds_ch_4"
cfg.slice_pad = 0.5


cfg.classes = ['Normal/Mild','Moderate','Severe']
cfg.n_classes = len(cfg.classes) # len(cfg.target_columns) * len(cfg.classes)


# OPTIMIZATION & SCHEDULE

cfg.fold = 0
cfg.epochs = 6

cfg.lr = 4e-4
cfg.optimizer = "AdamW"
cfg.weight_decay = 1e-4
cfg.warmup = 0.1 * cfg.epochs
cfg.batch_size = 8
cfg.batch_size_val=32
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 1.
cfg.num_workers = 8

#EVAL
cfg.post_process_pipeline = 'pp_dh_04'
cfg.metric = 'metric_dh_02' # 'default_metric
cfg.pp_bwd_fwd_fill = True
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
cfg.crop_buffer_ax = 1.
cfg.sequence_buffer_ratio_ax = 0.7
cfg.coords_logits_mask_thresh = 0.2
cfg.filter_logits_mask_thresh_trn = 0.7
cfg.filter_logits_mask_thresh_val = 0.7
cfg.max_closest_ax_distance = 3
cfg.ax_slice_range_clip = (5, 8)
cfg.filter_others_below = 0.5
cfg.filter_level_min_pred = 0.6



cfg.aug_seq_range = (0.5, 0.9)
# cfg.train_aug = None
# cfg.val_aug = None
cfg.train_aug = A.Compose([
    # A.Resize(cfg.crop_size,cfg.crop_size, interpolation = cv2.INTER_CUBIC),
#     A.D4(p=1),
    A.ShiftScaleRotate(shift_limit_x=0.2, shift_limit_y=0., scale_limit=0.2, rotate_limit=30, p=0.7),
#     #A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.75),
#     #A.InvertImg(p=0.5),
    A.CoarseDropout(num_holes_range=(10, 30),
                    hole_height_range=(int(0.01*cfg.img_size), int(0.08*cfg.img_size)),
                    hole_width_range=(int(0.01*cfg.img_size), int(0.08*cfg.img_size)), p=0.8),
    A.RandomCrop(width=24 * cfg.img_size//32, height=cfg.img_size, p=1.0),
    # A.Resize(cfg.img_size,cfg.img_size),
#     A.RandomCrop(height=96, width=192, p=1.0),
#    A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.5)
], additional_targets=dict((f'image{i}', 'image') for i in range(256)))
cfg.val_aug = None

#cfg.pretrained_weights = f"/mount/data/models/cfg_dh_12s1/fold0/checkpoint_last_seed99310.pth"
#cfg.pretrained_weights_strict = True
#cfg.pop_weights = None