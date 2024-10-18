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
    for k in 'input instance_numbers level_proba'.split():
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
idx = 500
class self:
    1
self = CustomDataset(df, cfg, aug = cfg.train_aug, mode = 'train')
# self = CustomDataset(df, cfg, aug = cfg.val_aug, mode = 'train')
# self = CustomDataset(df, cfg, aug = cfg.val_aug, mode = 'valid')


batch = [self.__getitem__(i) for i in range(0, len(self), len(self)//4)]
batch = tr_collate_fn(batch)
batch = batch_to_device(batch, 'cpu')
batch

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
        Sag
        '''
        ctrdf = pd.read_csv(cfg.loc_df)
        ctrdf = ctrdf[ctrdf.series_id.isin(self.df[self.df.series_description.str.contains('Sag')].series_id)]
        iddx = ctrdf.groupby('series_id')['logits_mask'].max()
        if self.mode == 'train':
            filter_mask_thresh = cfg.filter_logits_mask_thresh_trn
        else:
            filter_mask_thresh = cfg.filter_logits_mask_thresh_val

        ctrdf = ctrdf[ctrdf.series_id.isin(iddx[(iddx>filter_mask_thresh)].index)].reset_index(drop = True)

        logits_mask_thresh = cfg.coords_logits_mask_thresh
        ctrdf1 = ctrdf.query('logits_mask>@filter_mask_thresh').groupby('series_id    study_id  level'.split()).agg(
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
        self.ctrdf = self.ctrdf.drop(columns = ['x_mean', 'y_mean'])
        self.ctrdf = self.ctrdf.set_index('series_id')
        '''
        Ax
        '''
        locaxdf = pd.read_csv( cfg.locax_df )
        metadf = pd.read_pickle( cfg.meta_df )
        szdf = metadf['study_id series_id instance_number img_h img_w'.split()]\
            .drop_duplicates(subset = 'series_id instance_number'.split())

        locaxdf = pd.merge( locaxdf, szdf, on = 'series_id instance_number'.split(), how = 'left')
        loccols = "x_left  y_left  x_right  y_right".split()
        locaxdf[loccols] = locaxdf[loccols].values * locaxdf['img_w img_h img_w img_h'.split()].values
        locaxdf = locaxdf.set_index('series_id')



        mapaxdf = pd.read_pickle(cfg.mapax_df).rename(columns={'mapped_ax_instance_number':'instance_number_list'})
        mapaxdf['instance_number'] = mapaxdf['instance_number_list'].str[0]
        mapaxdf = mapaxdf[mapaxdf.distance.str[0]<cfg.max_closest_ax_distance]
        mapaxdf['levelohc'] = mapaxdf['level'].map({l:t for t,l in enumerate('l1_l2 l2_l3 l3_l4 l4_l5 l5_s1'.split())})
        mapaxdf = mapaxdf.sort_values('series_id level'.split()).reset_index(drop = True)
        buffers = {}
        for series_id, g in mapaxdf.groupby('series_id'):
            levelohcs = g.levelohc.values
            inums = g.instance_number.values
            iidx = levelohcs[1:]-levelohcs[:-1]==1
            diffs = (inums[1:]-inums[:-1])[iidx]
            if len(diffs)>0:
                buffers[series_id] = abs(np.median(diffs))
        mapaxdf['buffer'] = mapaxdf.series_id.map(buffers) * cfg.sequence_buffer_ratio_ax
        mapaxdf =  mapaxdf[~mapaxdf.buffer.isna()]

        mapaxdf['closest'] = mapaxdf.distance.str[0]
        mapaxdf = mapaxdf.sort_values('series_id  level closest'.split())
        mapaxdf = mapaxdf.drop_duplicates(subset='series_id level'.split(), keep = 'first')

        self.mapaxdf2 = []
        for r in  mapaxdf.itertuples():
            slc = slice(max(0,int(np.ceil(r.instance_number-r.buffer))),
                                int(np.floor(r.instance_number+r.buffer)))
            tmpdf = locaxdf.loc[r.series_id].set_index('instance_number').loc[slc].reset_index()
            xyxy = np.median( tmpdf[loccols].values, 0)
            inums = tmpdf.instance_number.tolist()
            dout = {c:getattr(r, c) for c in 'series_id level'.split()}
            dout['instance_number_list'] = inums
            dout['xyxy'] = xyxy.tolist()
            self.mapaxdf2.append(dout)
        self.mapaxdf2 = pd.DataFrame(self.mapaxdf2)
        self.mapaxdf2[['x_left','y_left','x_right','y_right']] = pd.DataFrame(self.mapaxdf2['xyxy'].values.tolist())

        self.mapaxdf2 =  self.mapaxdf2.drop(columns = 'xyxy')
        buffer =  \
            np.stack(((self.mapaxdf2.x_left - self.mapaxdf2.x_right).abs().values,
                     (self.mapaxdf2.y_left - self.mapaxdf2.y_right).abs().values)).max(0)
        buffer = (buffer *  (1 + cfg.crop_buffer_ax)).clip(40, 9999)
        x_ctr = self.mapaxdf2.filter(like='x_').mean(1)
        y_ctr = self.mapaxdf2.filter(like='y_').mean(1)
        self.mapaxdf2['x0'], self.mapaxdf2['y0'] = x_ctr - buffer // 2, y_ctr - buffer // 2
        self.mapaxdf2['x1'], self.mapaxdf2['y1'] = x_ctr + buffer // 2, y_ctr + buffer // 2

        self.mapaxdf2 = pd.merge(self.mapaxdf2,
                                 self.df['series_id study_id'.split()].drop_duplicates(),
                                 on = 'series_id', how = 'inner')


        '''
        Combine
        '''
        self.ctrdf = self.ctrdf.drop(columns = 'x_min  x_max  y_min  y_max '.split())
        self.mapaxdf2 = self.mapaxdf2.drop(columns = 'x_left  y_left  x_right  y_right'.split())


        self.ctrdf = pd.concat([self.ctrdf,self.mapaxdf2.set_index('series_id')])
        self.ctrdf["x0 y0 x1 y1".split()] = self.ctrdf["x0 y0 x1 y1".split()].clip(0, 9999999).astype(int)


        all_series = self.ctrdf.index.unique().tolist()
        self.df = self.df[self.df.series_id.isin (all_series)]
        # self.df = self.df.sample(frac = 1, replace = False)
        key_cols = 'series_id level'.split()
        keys = self.ctrdf.reset_index()[key_cols].drop_duplicates().reset_index(drop = True)
        self.df = pd.merge(self.df, keys, on = key_cols, how = 'inner')
        self.df_study_id = self.df['study_id level'.split()].drop_duplicates()
        self.ids = np.arange(len(self.df_study_id))
        self.df = self.df.set_index('study_id')


        self.meta = pd.read_pickle(cfg.meta_df).set_index('series_id').loc[all_series]
        self.meta.instance_number = self.meta.instance_number.astype(int)
        self.meta = self.meta.reset_index()\
            .sort_values('series_id instance_number'.split()).set_index('series_id')
        idx = 400

        self.mapper2 = {'nfn': 'neural_foraminal_narrowing',
                        'scs': 'spinal_canal_stenosis',
                        'ss': 'subarticular_stenosis'}


    def __getitem__(self, idx):

        # check_idx = []
        # for idx in range(len(self)):

        # idx = 2000
        study_id, level = self.df_study_id.iloc[idx]
        # if study_id not in CHKLS: continue
        rows = self.df.loc[[study_id]].query("level==@level").copy()
        feature_dicts_out = []

        for view in self.cfg.parts:#['nfn', 'scs', 'ss']:
            k = self.condition_map_rev[self.mapper2[view]]
            if k in rows.series_description.tolist():
                rows_v = rows.query("series_description==@k").copy()
                if self.mode == 'train':
                    rows_v = rows_v.sample(1)
                for _, row_v in rows_v.iterrows():
                    feature_dict = self.load_one(row_v, view = view)
                    feature_dict['condition'] = torch.tensor(self.cfg.condition_filter.index(k))
                    if view == 'scs':
                        feature_dict['target'] = torch.cat((feature_dict['target'], torch.tensor([-100])))
                    feature_dicts_out.append(feature_dict)

        '''
        input = feature_dict['input']
        Image.fromarray(input.to(torch.uint8).squeeze(1).reshape(-1, input.shape[-1]).numpy())
        '''
        #print([f['target'] for f in feature_dicts_out])
        # if feature_dicts_out[0]['target'][1]==-100:
        #     print([f['target'] for f in feature_dicts_out])
        #    break

        return feature_dicts_out

    def load_one(self, row_v, view):

        series_id = row_v.series_id
        label_key = self.cfg.condition_map[row_v['series_description']]

        row3 = self.ctrdf.loc[series_id ]
        row3 = row3.set_index('level').loc[row_v['level']]

        meta = self.meta.loc[series_id]
        ipp_ax = 0
        if view == 'ss': ipp_ax = 2
        ipp_0 = meta.ImagePositionPatient.apply(lambda x: x[ipp_ax]).tolist()
        if ( np.median(ipp_0[:3]) > np.median(ipp_0[-3:]) ):
            row3.instance_number_list = row3.instance_number_list[::-1][:]

        instance_numbers = torch.tensor(row3.instance_number_list)

        if self.mode=='train':
            prop = np.random.uniform(*self.cfg.aug_seq_range)
            seqlen = int(np.ceil(len(instance_numbers) * prop))
            startlim = len(instance_numbers) - seqlen
            if startlim>0:
                startpos = np.random.choice(startlim+1)
                instance_numbers = instance_numbers[startpos:seqlen+startpos]


        study_id = row3.study_id
        base_img_dir = f'{self.cfg.image_folder}/{study_id}/{series_id}'

        level_num = self.cfg.levels.index(row_v['level'])
        level_num = torch.tensor(level_num)

        label_cols = [i for i in self.label_cols if self.mapper2[view] in i]
        labels =  torch.from_numpy(row_v[label_cols].values.astype(int))

        inum = instance_numbers[len(instance_numbers)//2]
        img_name = f'{base_img_dir}/{inum}.jpeg'
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        x0, y0, x1, y1 = row3['x0 y0 x1 y1'.split()].clip(0, 99999).values

        # left_x > right_x
        # left -> top; right -> bottom .. after np.rot
        crops = []
        #imgls = []
        for t, inum in enumerate(instance_numbers):
            img_name = f'{base_img_dir}/{inum}.jpeg'
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            #imgls.append(img)
            crop = img[y0:y1,x0:x1]
            crop = cv2.resize(crop, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_CUBIC)
            if view in ['ss']: crop = np.rot90(crop)
            crops.append(crop)

        # Image.fromarray(np.concatenate(crops))
        '''
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        crop = img[y0:y1,x0:x1]
        crop = cv2.resize(crop, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_CUBIC)
        crop = np.rot90(crop)
        imgout = self.aug(image = crop)['image']
        Image.fromarray( crop)
        Image.fromarray( imgout)
        '''

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
            "study_id":  torch.tensor(study_id),
            'level_num': level_num,

            'instance_numbers': instance_numbers,
        }
        return feature_dict

    def __len__(self):
        return len(self.ids)

    def augment(self, img):
#         img = img.astype(np.float32)
        transformed = self.aug(image=img)
        trans_img = transformed["image"]
        return trans_img

