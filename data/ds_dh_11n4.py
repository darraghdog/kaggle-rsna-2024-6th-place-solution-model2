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
try:
    from bounding_box import bounding_box as bbfn
except:
    print('bounding_box not installed')

from torch.nn.utils.rnn import pad_sequence


def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def collate_fn(batch):
    # Remove error reads
    batch = [b for b in batch if b is not None]

    batchout = {}
    for k in 'input target target_mask instance_numbers'.split():
        if k in batch[0]:
            batchout[k] = pad_sequence([b[k] for b in batch],  batch_first=True)

    batchout['mask'] = pad_sequence([torch.ones_like(b['instance_numbers']) for b in batch],  batch_first=True)

    for k in 'series_id'.split():
        if k in batch[0]:
            batchout[k] = torch.stack([b[k] for b in batch])

    return batchout


tr_collate_fn = collate_fn
val_collate_fn = collate_fn


def set_pandas_display():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows',10000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
set_pandas_display()




'''

df = pd.read_csv(cfg.train_df)
df = df.query('fold!=1')

aug = cfg.train_aug
#aug = cfg.val_aug
mode="valid"
idx = 10
class self:
    1
self = CustomDataset(df, cfg, aug = cfg.train_aug, mode = 'train')
self = CustomDataset(df, cfg, aug = cfg.val_aug, mode = 'valid')


self = CustomDataset(df, cfg, aug = cfg.val_aug, mode = 'test')


batch = [self.__getitem__(i) for i in range(0, 3500, 1200)]
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
        self.df = df['study_id series_id fold series_description'.split()].copy()
        self.df = self.df[self.df.series_description.isin(cfg.condition_filter)]

        self.aug = aug
        self.data_folder = cfg.data_folder

        '''
        Anno prediction
        '''
        if mode!='test':
            coorddf = pd.read_csv(cfg.coord_df, usecols = 'study_id series_id instance_number condition level'.split())
            coorddf['condition'] = coorddf.condition.str.replace('Neural Foraminal Narrowing', 'nfn')
            coorddf['condition'] = coorddf.condition.str.replace('Spinal Canal Stenosis', 'scs')
            coorddf['condition'] = coorddf.condition.str.lower().str.replace(' ', '_')
            coorddf = coorddf[coorddf.condition.isin('left_nfn right_nfn scs'.split())]
            coorddf = coorddf.groupby('study_id series_id instance_number condition'.split())\
                            ['level'].count().reset_index().rename(columns = {'level':'count'})
            coorddf = coorddf.pivot(index='study_id series_id instance_number'.split(), \
                                    columns='condition', values='count').fillna(0).reset_index()
            coorddf['has_label'] = 1

        trncdfls = []
        if mode=='train':
            ddf = coorddf['study_id series_id'.split()].drop_duplicates()
        else:
            ddf = self.df['study_id series_id'.split()].drop_duplicates()

        for row in tqdm(ddf.itertuples(), total = len(ddf)):
            dnms = glob.glob(f'{self.cfg.dicom_folder}/{row.study_id}/{row.series_id}/*.dcm')
            row_out = row._asdict()
            row_out['instance_number'] = sorted([int(i.split('/')[-1].split('.')[0]) for i in dnms])
            del row_out['Index']
            trncdfls.append(row_out)
        ddf = pd.DataFrame(trncdfls).explode('instance_number')

        if mode!='test':
            coorddf = pd.merge(ddf, coorddf, on = "study_id series_id instance_number".split(), how = 'left')
            coorddf["left_nfn  right_nfn  scs has_label".split()] = coorddf["left_nfn  right_nfn  scs has_label".split()].fillna(0)
            coorddf = coorddf.sort_values("series_id  instance_number".split()).reset_index(drop = True)
            coorddf['has_label'] = coorddf.groupby('series_id')['has_label'].transform('max')
        else:
            coorddf = ddf.copy()
            coorddf['left_nfn right_nfn scs has_label'.split()] = 0.
            coorddf = coorddf.sort_values("series_id  instance_number".split()).reset_index(drop = True)

        self.coorddf = coorddf.set_index('series_id')

        '''
        Location
        '''
        locdf = pd.read_csv(cfg.loc_df)
        locdf = locdf.groupby('series_id study_id level'.split())['x y'.split()].mean().reset_index()
        locdf = \
            pd.merge(locdf.groupby('series_id')['x y'.split()].min().rename(columns = {'x':'x0', 'y':'y0'}).reset_index(), \
                       locdf.groupby('series_id')['x y'.split()].max().rename(columns = {'x':'x1', 'y':'y1'}).reset_index(), on='series_id')
        locdf['xdiff'], locdf['ydiff'] = (locdf['x1']-locdf['x0']), (locdf['y1']-locdf['y0'])
        locdf['maxdiff'] = locdf['xdiff ydiff'.split()].max(1)

        locdf['x0'], locdf['x1'] =  locdf['x0 x1'.split()].mean(1) - locdf['maxdiff'] * 0.7, \
                                    locdf['x0 x1'.split()].mean(1) + locdf['maxdiff'] * 0.7
        locdf['y0'], locdf['y1'] =  locdf['y0 y1'.split()].mean(1) - locdf['maxdiff'] * 0.7, \
                                    locdf['y0 y1'.split()].mean(1) + locdf['maxdiff'] * 0.7
        locdf  = locdf['series_id x0 y0 x1 y1'.split()].clip(0, 9999999999).round().astype(int)
        self.locdf = locdf.copy().set_index('series_id')

        iddf = self.coorddf.reset_index()['series_id study_id'.split()].drop_duplicates()

        iddf = iddf[iddf.series_id.isin(self.df.series_id)].reset_index(drop = True)

        self.ids = iddf.sort_values('study_id')['series_id'].values
        print(f"Number of crops : {len(self.coorddf)}")
        print(f"Number of series : {len(self.ids)}")

        idx = 0

        self.meta = pd.read_pickle(cfg.meta_df)#.set_index('series_id').loc[all_series]
        self.meta.instance_number = self.meta.instance_number.astype(int)
        self.meta = self.meta[' series_id ImagePositionPatient instance_number PixelSpacing  SpacingBetweenSlices SliceThickness  SliceLocation'.split()]
        self.meta = self.meta.sort_values('series_id instance_number'.split()).set_index('series_id')

    def __getitem__(self, idx):

        # for idx in np.arange(0, 20000, 40):
        series_id = self.ids[idx]
        rows = self.coorddf.loc[series_id ]

        study_id = int(rows.iloc[0].study_id)
        instance_numbers = rows.instance_number.tolist()

        # Load image
        inum = instance_numbers[len(instance_numbers)//2]
        center_img_name = f'{self.cfg.image_folder}/{study_id}/{series_id}/{inum}.jpeg'
        assert os.path.isfile(center_img_name)
        center_img = cv2.imread(center_img_name, cv2.IMREAD_GRAYSCALE)

        meta = self.meta.loc[series_id].set_index('instance_number').loc[instance_numbers]
        ipp_0 = meta.ImagePositionPatient.apply(lambda x: x[0]).tolist()
        meta.loc[:,'ipp_0'] = ipp_0

        if ( np.median(ipp_0[:3]) > np.median(ipp_0[-3:]) ):
            instance_numbers = instance_numbers[::-1][:]

        bbox = self.locdf.loc[series_id]

        imgls = []
        for inum in instance_numbers:
            img_name = f'{self.cfg.image_folder}/{study_id}/{series_id}/{int(inum)}.jpeg'
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            img = img[bbox.y0:bbox.y1, bbox.x0:bbox.x1]
            if img.shape[0]!=center_img.shape[0]:
                img = cv2.resize(img, (center_img.shape[1], center_img.shape[0]), interpolation = cv2.INTER_CUBIC)
            imgls.append(img)

        # Image.fromarray(np.concatenate(imgls))

        if self.aug:
            aug_input = dict((f'image{t}', i) for t,i in enumerate(imgls))
            aug_input['image'] = imgls[0]
            aug_output = self.aug (**aug_input)
            imgls = [aug_output[f'image{t}'] for t,i in enumerate(imgls)]

        imgls = np.stack(imgls)[:,None]
        torch_img = torch.tensor(imgls).float()#.permute(0, 3, 1, 2)


        label_cond_rows =  self.coorddf.loc[series_id].set_index('instance_number').loc[instance_numbers]
        label_cond =  label_cond_rows[self.cfg.classes].values
        mask_cond =  np.zeros_like(label_cond).astype(int)
        if label_cond_rows.has_label.max()>0:
            mask_cond[:,label_cond.sum(0)>0] = 1

        feature_dict = {
            "input": torch_img,
            "target": torch.tensor(label_cond).float(),
            "target_mask": torch.tensor(mask_cond).long(),
            'series_id': torch.tensor(series_id),
            'instance_numbers': torch.tensor(instance_numbers),
        }
        return feature_dict

    def __len__(self):
        return len(self.ids)

    def load_one(self, image_id):
        fp = f'{self.data_folder}{image_id}.npy'
        try:
            img = np.load(fp).transpose(1,2,0)
        except Exception as e:
            print(e)
        return img

    def augment(self, img):
#         img = img.astype(np.float32)
        transformed = self.aug(image=img)
        trans_img = transformed["image"]
        return trans_img


