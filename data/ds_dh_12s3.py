import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import pandas as pd
import pickle
import pydicom
from tqdm import tqdm
from PIL import Image
# from bounding_box import bounding_box as bbfn
from torch.nn.utils.rnn import pad_sequence
import warnings

def set_pandas_display():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows',10000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
set_pandas_display()

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


#tr_collate_fn = torch.utils.data.default_collate
#val_collate_fn = torch.utils.data.default_collate

def collate_fn(batch):
    # Remove error reads
    batch = sum(batch, [])
    batch = [b for b in batch if b is not None]

    batchout = {}
    for k in 'input instance_numbers slicelocation'.split():
        if k in batch[0]:
            batchout[k] = pad_sequence([b[k] for b in batch],  batch_first=True)

    batchout['mask'] = pad_sequence([torch.ones_like(b['instance_numbers']) for b in batch],  batch_first=True)

    for k in 'study_id series_id target level_num condition'.split():
        if k in batch[0]:
            batchout[k] = torch.stack([b[k] for b in batch])

    return batchout


tr_collate_fn = collate_fn
val_collate_fn = collate_fn



'''

df = pd.read_csv(cfg.train_df)
df = df.query('fold!=0')

aug = cfg.train_aug
cfg.val_aug
mode="valid"
idx = 10
class self:
    1
self = CustomDataset(df, cfg, aug = cfg.train_aug, mode = 'train')
# self = CustomDataset(df, cfg, aug = cfg.val_aug, mode = 'valid')


batch = [self.__getitem__(i) for i in range(0,  len(self), len(self)//6)]
batch = tr_collate_fn(batch)
batch = batch_to_device(batch, 'cpu')


'''


def read_dcm_ret_arr(src_path):
    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    return image

class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.df = self.df[self.df.series_description.isin(cfg.condition_filter)]
        label2id = {'Normal/Mild': 0, 'Moderate':1, 'Severe':2}

        self.df = self.df.fillna(-100)
        self.df = self.df.replace(label2id)

        tmpdfs = []
        self.label_cols = [s.replace('_l1_l2', "") for s in cfg.targets[0::5]]

        self.label_cols_nfn = [i for i in self.label_cols if 'foraminal' in i]
        self.label_cols_ss = [i for i in self.label_cols if 'spinal_canal' in i]
        self.condition_map_rev = {v:k for k,v in cfg.condition_map.items()}

        for tt,i in enumerate(cfg.levels):
            cols = cfg.targets[tt::5] + "fold study_id series_id series_description".split()
            tmpdf = self.df[cols].copy()
            tmpdf.loc[:,'level'] = i
            tmpdf.columns = self.label_cols + tmpdf.columns[len(cfg.targets)//len(cfg.levels):].tolist()
            tmpdfs.append(tmpdf)
        self.df = pd.concat(tmpdfs)

        self.aug = aug
        self.data_folder = cfg.data_folder

        '''
        Bounding boxes

        cfg.loc_df = f'{cfg.data_dir}/sag_xy/shifted_spinenet_v01.csv.gz'
        cfg.loc_df = f'{cfg.data_dir}/sag_xy/test__cfg_dh_14f_locsag_test.csv.gz'
        '''
        ctrdf = pd.read_csv(cfg.loc_df)
        ctrdf = ctrdf[ctrdf.series_id.isin(self.df.series_id)]
        iddx = ctrdf.groupby('series_id')['logits_mask'].max()
        ctrdf = ctrdf[ctrdf.series_id.isin(iddx[(iddx>0.8)].index)].reset_index(drop = True)

        # iddx = ctrdf['logits_mask'].values == ctrdf.groupby('series_id')['logits_mask'].transform(max).values


        logits_mask_thresh = cfg.coords_logits_mask_thresh
        ctrdf1 = ctrdf.query('logits_mask>0.8').groupby('series_id    study_id  level'.split()).agg(
                    x_min=('x', 'min'),
                    x_max=('x', 'max'),
                    y_min=('y', 'min'),
                    y_max=('y', 'max'),
                ).reset_index()
        ctrdf2 = ctrdf.query('logits_mask>@logits_mask_thresh').groupby('series_id    study_id  level'.split()).agg(
                    instance_number_list=('instance_number', lambda x: sorted(list(x)))
                ).reset_index()

        self.ctrdf = pd.merge(ctrdf1, ctrdf2, on = 'series_id    study_id  level'.split(), how = 'inner')
        self.ctrdf['x_mean'], self.ctrdf['y_mean'] = self.ctrdf.filter(like='x_m').mean(1), self.ctrdf.filter(like='y_m').mean(1)
        self.ctrdf = self.ctrdf.sort_values('series_id level'.split()).reset_index(drop = True)
        self.ctrdf['x0 y0'.split()] = self.ctrdf.groupby(['series_id']).apply(lambda x: x[['x_mean', 'y_mean']].shift(1)).values
        self.ctrdf['x1 y1'.split()] = self.ctrdf.groupby(['series_id']).apply(lambda x: x[['x_mean', 'y_mean']].shift(-1)).values

        diff1 = ((self.ctrdf.x_mean - self.ctrdf.x0) ** 2 + (self.ctrdf.y_mean - self.ctrdf.y0) ** 2) ** 0.5
        diff2 = ((self.ctrdf.x_mean - self.ctrdf.x1) ** 2 + (self.ctrdf.y_mean - self.ctrdf.y1) ** 2) ** 0.5
        buffer = np.stack((diff1.fillna(0).values, diff2.fillna(0).values)).max(0)
        buffer = buffer * (1 + cfg.crop_buffer )
        buffer = buffer.clip(40, 99999)
        self.ctrdf['x0'] = np.floor(self.ctrdf.x_mean - buffer ).astype(int)
        self.ctrdf['x1'] = np.ceil(self.ctrdf.x_mean + buffer ).astype(int)
        self.ctrdf['y0'] = np.floor(self.ctrdf.y_mean - buffer ).astype(int)
        self.ctrdf['y1'] = np.ceil(self.ctrdf.y_mean + buffer ).astype(int)
        self.ctrdf['x0 y0 x1 y1'.split()] = self.ctrdf['x0 y0 x1 y1'.split()].clip(0, 99999)
        self.ctrdf = self.ctrdf.drop(columns = 'x_mean y_mean x_min y_min x_max y_max'.split())

        if self.mode == 'train':
            # Filter out where annotation does not match
            coorddf = pd.read_csv(cfg.coord_df)
            coorddf['level'] = coorddf['level'].str.replace('/','_').str.lower()
            coorddf['condition'] = coorddf.condition.str.replace(' ','_').str.lower()
            coorddf = coorddf['series_id level condition x y'.split()]
            chkdf = pd.merge(self.ctrdf.drop(columns = 'instance_number_list study_id'.split()),
                             coorddf, how = 'left', on = 'series_id level'.split())
            chkdf['x_lbld'] = (chkdf['x'] - chkdf['x0']) / (chkdf['x1'] - chkdf['x0'])
            chkdf['y_lbld'] = (chkdf['y'] - chkdf['y0']) / (chkdf['y1'] - chkdf['y0'])
            chkdf['drop_list'] = (chkdf['y_lbld'] != chkdf['y_lbld'].clip(0.2, 0.8)) | \
                                (chkdf['x_lbld'] != chkdf['x_lbld'].clip(0.2, 0.8))
            print(f'Keep ->>> \n{chkdf["drop_list"].value_counts()}')
            chkdf = chkdf[chkdf['drop_list']].dropna()
            remove_list = []
            # For nfn if over 6 are not matching remove from self.ctrdf
            nfn_miss_ctr = chkdf[chkdf.condition.str.contains('foraminal')].groupby('series_id')['level'].count().sort_values()
            remove_list += nfn_miss_ctr[nfn_miss_ctr>6].index.tolist()
            # For scs if over 3 are not matching remove from self.ctrdf
            scs_miss_ctr = chkdf[chkdf.condition.str.contains('stenosis')].groupby('series_id')['level'].count().sort_values()
            remove_list += scs_miss_ctr[scs_miss_ctr>3].index.tolist()
            # Remove these fully
            self.ctrdf = self.ctrdf[~self.ctrdf['series_id'].isin(remove_list)]
            # For the others, remove loss for the given level
            self.df = self.df.reset_index(drop = True)
            for r in chkdf.itertuples():
                drop_pos = self.df[(r.series_id==self.df.series_id) & (r.level==self.df.level) ].index
                assert len(drop_pos) == 1
                drop_row = self.df.loc[drop_pos[0]]
                assert drop_row.series_id == r.series_id
                assert drop_row.level == r.level
                self.df.loc[drop_pos[0], r.condition] = -100
                #print(r.series_id, r.level, r.condition)

        self.ctrdf = self.ctrdf.set_index('series_id')

        all_series = self.ctrdf.index.unique().tolist()

        self.df = self.df[self.df.series_id.isin (all_series)]
        # self.df = self.df.sample(frac = 1, replace = False)
        key_cols = 'series_id level'.split()
        keys = self.ctrdf.reset_index()[key_cols].drop_duplicates().reset_index(drop = True)
        self.df = pd.merge(self.df, keys, on = key_cols, how = 'inner')

        self.df_study_id = self.df['study_id level'.split()].drop_duplicates()
        self.ids = np.arange(len(self.df_study_id))
        self.df = self.df.set_index('study_id')


        self.meta = pd.read_pickle(cfg.meta_df)#.set_index('series_id').loc[all_series]
        self.meta.instance_number = self.meta.instance_number.astype(int)
        self.meta = self.meta[' series_id ImagePositionPatient instance_number PixelSpacing  SpacingBetweenSlices SliceThickness  SliceLocation'.split()]
        self.meta = self.meta.sort_values('series_id instance_number'.split())
        self.meta = self.meta.set_index('series_id').loc[all_series]


        idx = 5000
        #trncdf = pd.read_csv(cfg.coord_df)


    def __getitem__(self, idx):

        # check_idx = []
        # for idx in range(len(self)):
        study_id, level = self.df_study_id.iloc[idx]
        rows = self.df.loc[[study_id]].query("level==@level").copy()
        feature_dicts_out = []

        # neural_foraminal_narrowing
        k = self.condition_map_rev['neural_foraminal_narrowing']
        if k in rows.series_description.tolist():
            rows_nfn = rows.query("series_description==@k").copy()
            if self.mode == 'train':
                rows_nfn = rows_nfn.sample(1)
            for _, row_nfn in rows_nfn.iterrows():
                feature_dict_nfn = self.load_one(row_nfn, neural_foraminal_narrowing = True)
                feature_dict_nfn['condition'] = torch.tensor(self.cfg.condition_filter.index(k))
                feature_dicts_out.append(feature_dict_nfn)

        # spinal_canal_stenosis
        k = self.condition_map_rev['spinal_canal_stenosis']
        if k in rows.series_description.tolist():
            rows_ss = rows.query("series_description==@k").copy()
            if self.mode == 'train':
                rows_ss = rows_ss.sample(1)
            for _, row_ss in rows_ss.iterrows():

                feature_dict_ss = self.load_one(row_ss, neural_foraminal_narrowing = False)
                feature_dict_ss['condition'] = torch.tensor(self.cfg.condition_filter.index(k)).long()
                feature_dict_ss['target'] = torch.cat((feature_dict_ss['target'], torch.tensor([-100])))
                feature_dicts_out.append(feature_dict_ss)

        '''
        row2 = row_nfn
        row2['level'] = 'l5_s1'
        neural_foraminal_narrowing = True
        '''

        return feature_dicts_out

    def load_one(self, row2, neural_foraminal_narrowing = False):

        series_id = row2.series_id
        label_key = self.cfg.condition_map[row2['series_description']]

        row3 = self.ctrdf.loc[series_id ]
        row3 = row3.set_index('level').loc[row2['level']]

        meta = self.meta.loc[series_id].copy()
        ipp_0 = meta.ImagePositionPatient.apply(lambda x: x[0]).tolist()
        meta.loc[:,'ipp_0'] = ipp_0

        level_num = self.cfg.levels.index(row2['level'])
        level_num = torch.tensor(level_num)
        label_cols = self.label_cols_nfn if neural_foraminal_narrowing \
                            else self.label_cols_ss
        labels =  torch.from_numpy(row2[label_cols].values.astype(int))

        if ( np.median(ipp_0[:3]) > np.median(ipp_0[-3:]) ):
            row3.instance_number_list = row3.instance_number_list[::-1][:]

        if neural_foraminal_narrowing:
            if (np.random.random()>0.5) and (self.mode == 'train'):
                row3.instance_number_list = row3.instance_number_list[::-1][:]
                labels  = torch.flip(labels,[0]).clone()

        instance_numbers = torch.tensor(row3.instance_number_list)

        study_id = row3.study_id
        base_img_dir = f'{self.cfg.image_folder}/{study_id}/{series_id}'

        meta = meta.set_index('instance_number').loc[instance_numbers.tolist()]
        #slicelocation = meta.SliceLocation.clip(-self.cfg.slicelocation_clip,
        #                                        self.cfg.slicelocation_clip).values + self.cfg.slicelocation_clip
        slicelocation = meta.ipp_0.clip(-self.cfg.slicelocation_clip,
                                                self.cfg.slicelocation_clip).values + self.cfg.slicelocation_clip
        slicelocation = torch.from_numpy(slicelocation).round().long()


        inum = instance_numbers[len(instance_numbers)//2]
        img_name = f'{base_img_dir}/{inum}.jpeg'
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        x0, y0, x1, y1 = row3['x0 y0 x1 y1'.split()].clip(0, 99999).values

        crops = []
        for t, inum in enumerate(instance_numbers):
            img_name = f'{base_img_dir}/{inum}.jpeg'
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            crop = img[y0:y1,x0:x1]
            crop = cv2.resize(crop, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_CUBIC)
            crops.append(crop)

        # Image.fromarray(np.concatenate(crops))

        if self.aug:
            aug_input = dict((f'image{t}', i) for t,i in enumerate(crops))
            aug_input['image'] = crops[0]
            aug_output = self.aug (**aug_input)
            crops = [aug_output[f'image{t}'] for t,i in enumerate(crops)]

        crops = np.stack(crops)[:,None]
        torch_img = torch.tensor(crops).float()#.permute(0, 3, 1, 2)

        feature_dict = {
            "input": torch_img,
            "target": labels,
            "series_id":  torch.tensor(series_id),
            "slicelocation": slicelocation,
            "study_id":  torch.tensor(study_id),
            'level_num': level_num,
            'instance_numbers': instance_numbers,
        }
        return feature_dict

    def __len__(self):
        return len(self.ids)


    def get_bbox(self, row):
        # Crop image
        bbox = row['x0 y0 x1 y1'.split()].values
        crop_x_dim, crop_y_dim  = row.x1-row.x0, row.y1-row.y0

        bbox[0] = bbox[0] - (self.cfg.crop_buffer * crop_x_dim)
        bbox[2] = bbox[2] + (self.cfg.crop_buffer * crop_x_dim)
        bbox[1] = bbox[1] - (self.cfg.crop_buffer * crop_y_dim)
        bbox[3] = bbox[3] + (self.cfg.crop_buffer * crop_y_dim)

        # Make crop a square
        crop_x_dim2, crop_y_dim2  = bbox[2]-bbox[0], bbox[3]-bbox[1]
        if crop_x_dim2 < crop_y_dim2:
            bbox[0] = bbox[0] - (crop_y_dim2 - crop_x_dim2)/2
            bbox[2] = bbox[2] + (crop_y_dim2 - crop_x_dim2)/2
        else:
            bbox[1] = bbox[1] - (crop_x_dim2 - crop_y_dim2)/2
            bbox[3] = bbox[2] + (crop_x_dim2 - crop_y_dim2)/2
        bbox = bbox.astype(np.float32).round().astype(int)

        return bbox

    def augment(self, img):
#         img = img.astype(np.float32)
        transformed = self.aug(image=img)
        trans_img = transformed["image"]
        return trans_img
