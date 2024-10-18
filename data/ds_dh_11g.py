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
#aug = cfg.val_aug
mode="valid"
idx = 10
class self:
    1
self = CustomDataset(df, cfg, aug = cfg.train_aug, mode = 'train')
# self = CustomDataset(df, cfg, aug = cfg.val_aug, mode = 'valid')

batch = [self.__getitem__(i) for i in range(0, 50000, 3000)]
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
        Shifted and filtered spinenet labels
        '''
        trndfdict = self.df.set_index('series_id')['study_id'].transpose().to_dict()
        trncdf = pd.read_csv(cfg.coord_df, usecols = 'series_id instance_number level x y'.split())
        trncdf = trncdf[trncdf.series_id.isin(self.df.series_id)]
        trncdf = trncdf[trncdf.level.isin(cfg.levels)].reset_index(drop = True)
        grpcols = 'study_id series_id instance_number'.split()
        trncdf['study_id'] = trncdf['series_id'].map(trndfdict)
        trncdf = trncdf.groupby(grpcols + ["level"])['x y'.split()].mean().reset_index()

        trncdf.set_index(["series_id", "instance_number", "level", "study_id"], inplace=True)
        trncdf = trncdf.unstack("level")
        trncdf.columns = [f"{x}__{y}" for x, y in trncdf.columns]
        trncdf.reset_index(inplace=True)

        self.trncdf = trncdf.fillna(-100)
        self.trncdf.columns = self.trncdf.columns.str.lower()
        self.trncdf['masked'] = False

        '''
        Add mask for images which hanve no label
        '''
        if cfg.full_test_set:
            trncdftst = df['study_id series_id'.split()].drop_duplicates().copy()
        else:
            trncdftst = self.trncdf['study_id series_id'.split()].drop_duplicates().copy()
        trncdfls = []
        print('Get all series instance numbers with glob : ')
        for row in tqdm(trncdftst.itertuples(), total = len(trncdftst)):
            dnms = glob.glob(f'{self.cfg.dicom_folder}/{row.study_id}/{row.series_id}/*.dcm')
            row_out = row._asdict()
            row_out['instance_number'] = sorted([int(i.split('/')[-1].split('.')[0]) for i in dnms])
            del row_out['Index']
            trncdfls.append(row_out)
        trncdf = pd.DataFrame(trncdfls).explode('instance_number').reset_index(drop = True)
        for c,v in zip(self.cfg.classes, np.arange(len(self.cfg.classes))): trncdf[c] = -100
        trncdf['masked'] = True

        self.trncdf = pd.concat([self.trncdf, trncdf]) \
            .drop_duplicates(subset = ['study_id', 'series_id', 'instance_number'], keep = 'first')

        self.ids = self.trncdf['series_id'].values
        self.trncdf = self.trncdf.set_index('series_id')
        self.trncdf = self.trncdf.sample(frac = 1, replace = False)
        print(f"Number of crops : {len(self.trncdf)}")
        idx = 0

        self.inums = self.trncdf.reset_index().groupby('series_id')['instance_number'].apply(lambda x: sorted(x.tolist()))

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


        instance_numbers = self.inums.loc[series_id]
        padding = 10
        instance_numbers_padded = ([-1] * padding + instance_numbers + [-1] * padding)
        from_,to_  = (instance_numbers_padded.index(instance_number) - self.cfg.in_channels // 2), (instance_numbers_padded.index(instance_number) + 1 + self.cfg.in_channels // 2)
        instance_numbers_load = instance_numbers_padded[from_:to_]
        imgls = []
        for inum in instance_numbers_load:
            if inum > -1:
                img_name = f'{self.cfg.image_folder}/{study_id}/{series_id}/{int(inum)}.jpeg'
                assert os.path.isfile(img_name)
                img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            else:
                img = np.zeros_like(center_img)
            if img.shape[0]!=center_img.shape[0]:
                img = cv2.resize(img, (center_img.shape[1], center_img.shape[0]), interpolation = cv2.INTER_CUBIC)
            imgls.append(img)
        img = cv2.merge(imgls)


        label = row[self.cfg.classes].values.astype(np.float32).copy()
        label_mask = (label!=-100).astype(int)
        label[label_mask==0] = center_img.shape[0]//2
        target_mask = torch.tensor((label_mask.sum()/ label_mask.shape[0])).float()
        target_mask = target_mask.clip(0.01, 0.99)

        '''
        viz_img1 = center_img[:,:,None][:,:,[0,0,0]].copy()
        bbox = [*label[:2]-5, *label[:2]+5]
        bbfn.add(viz_img1, *list(map(int, bbox)), color = 'yellow')
        bbox = [*label[-2:]-5, *label[-2:]+5]
        bbfn.add(viz_img1, *list(map(int, bbox)), color = 'green')
        Image.fromarray(viz_img1)


        viz_img1 = axial_img[:,:,None][:,:,[0,0,0]].copy()
        bbox = [*axial_kps[:2]-5, *axial_kps[:2]+5]
        print(bbox)
        bbfn.add(viz_img1, *list(map(int, bbox)), color = 'yellow')
        bbox = [*axial_kps[-2:]-5, *axial_kps[-2:]+5]
        print(bbox)
        bbfn.add(viz_img1, *list(map(int, bbox)), color = 'green')
        Image.fromarray(viz_img1)
        '''

        keep_running = True
        n_kp_pairs = len(self.cfg.classes)//2
        while keep_running:
            xy_pairs = torch.from_numpy(label.clip(0,999999)).reshape(n_kp_pairs, 2)
            if self.aug:
                try:
                    aug_out = self.aug(image=img, keypoints = xy_pairs.numpy())
                    axial_kps = np.array(aug_out['keypoints']).flatten()
                    axial_img = aug_out['image']
                except:
                    print('Hitting the ds_dh_11g except')
                    axial_img = cv2.resize(img, (self.cfg.img_size,self.cfg.img_size), interpolation=cv2.INTER_CUBIC)
                    axial_kps = xy_pairs.flatten().numpy() * self.cfg.img_size /  img.shape[0]
                if len(axial_kps)== n_kp_pairs * 2: keep_running = False

        xy_size = list(axial_img.shape)[:2][::-1]
        label = axial_kps / np.array(xy_size*n_kp_pairs)
        torch_img = torch.tensor(axial_img).permute(2, 0, 1)


        feature_dict = {
            "input": torch_img,
            "input_mask": target_mask ,
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


