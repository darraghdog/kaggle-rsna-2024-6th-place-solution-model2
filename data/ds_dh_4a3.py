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

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


tr_collate_fn = torch.utils.data.default_collate
val_collate_fn = torch.utils.data.default_collate


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

for i in range(0, 10000, 500): self.__getitem__(i)
batch = [self.__getitem__(i) for i in range(0, 10000, 500)]
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
        Dicom level human labels.
        '''
        trndfdict = self.df.set_index('series_id')['study_id'].transpose().to_dict()
        trncdf = pd.read_csv(cfg.coord_df, usecols = 'series_id instance_number condition x y'.split())
        trncdf = trncdf[trncdf.series_id.isin(self.df.series_id)].reset_index(drop = True)
        trncdf['side'] = trncdf.condition.str.split().str[0]
        trncdf = trncdf.drop(columns = "condition")
        trncdf['study_id'] = trncdf['series_id'].map(trndfdict)

        trncdf.set_index(["series_id", "instance_number", "side", "study_id"], inplace=True)
        trncdf = trncdf.unstack("side")
        trncdf.columns = [f"{x}_{y}" for x, y in trncdf.columns]
        trncdf.reset_index(inplace=True)

        self.trncdf = trncdf.fillna(-100)
        self.trncdf.columns = self.trncdf.columns.str.lower()

        if (getattr(cfg, 'full_test_set', False) & cfg.val):
            trncdftst = self.df['study_id series_id'.split()].drop_duplicates()
            trncdfls = []
            for row in trncdftst.itertuples():
                dnms = glob.glob(f'{self.cfg.dicom_folder}/{row.study_id}/{row.series_id}/*.dcm')
                row_out = row._asdict()
                row_out['instance_number'] = sorted([int(i.split('/')[-1].split('.')[0]) for i in dnms])
                del row_out['Index']
                trncdfls.append(row_out)
            trncdf = pd.DataFrame(trncdfls).explode('instance_number').reset_index(drop = True)
            for c,v in zip(self.cfg.classes, [50.,60.,70.,80.]): trncdf[c] = v
            trncdf[self.cfg.classes]
            self.trncdf = trncdf


        self.ids = self.trncdf['series_id'].values
        self.trncdf = self.trncdf.set_index('series_id')
        print(f"Number of crops : {len(self.trncdf)}")

        idx = 1003

    def __getitem__(self, idx):

        # for idx in np.arange(0, 20000, 40):
        row = self.trncdf.iloc[idx]

        series_id = int(row.name)
        study_id = int(row.study_id)
        instance_number = int(row.instance_number)

        # Load image
        center_img_name = f'{self.cfg.image_folder}/{study_id}/{series_id}/{int(instance_number)}.jpeg'
        assert os.path.isfile(center_img_name)
        center_img = cv2.imread(center_img_name, cv2.IMREAD_GRAYSCALE)

        label = row[self.cfg.classes].values.copy()
        if (getattr(self.cfg, 'full_test_set', False) & self.cfg.val):
            label = label.astype(np.float32)

        '''
        viz_img1 = center_img[:,:,None][:,:,[0,0,0]].copy()
        bbox = [*label[:2]-5, *label[:2]+5]
        bbfn.add(viz_img1, *list(map(int, bbox)), color = 'yellow')
        Image.fromarray(viz_img1)


        viz_img1 = axial_img[:,:,None][:,:,[0,0,0]].copy()
        bbox = [*axial_kps[:2]-5, *axial_kps[:2]+5]
        bbfn.add(viz_img1, *list(map(int, bbox)), color = 'yellow')
        Image.fromarray(viz_img1)
        '''
        label_mask = (label!=-100).astype(int)
        label[label_mask==0] = center_img.shape[0]//2

        keep_running = True
        while keep_running:
            if self.aug:
                aug_out = self.aug(image=center_img,
                                   keypoints = torch.from_numpy(label.clip(0,999999)).reshape(2, -1).numpy())
                axial_kps = np.array(aug_out['keypoints']).flatten()
                axial_img = aug_out['image']
                if len(axial_kps)==4: keep_running = False

        xy_size = list(axial_img.shape)[::-1]
        label = axial_kps / np.array(xy_size*2)
        torch_img = torch.tensor(axial_img)

        feature_dict = {
            "input": torch_img,
            "target": torch.tensor(label).float(),
            "target_mask": torch.tensor(label_mask).long(),
            'series_id': torch.tensor(series_id),
            'instance_numbers': torch.tensor(instance_number),
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


