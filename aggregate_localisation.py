import os
import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import set_pandas_display
set_pandas_display()


metadf = pd.read_pickle('datamount/meta_v1.pkl')
os.makedirs('datamount/sag_xy/', exist_ok = True)
os.makedirs('datamount/axial_xy/', exist_ok = True)


'''
cfg_dh_05b5_loc_test2
'''

tmpdfls = []
for FOLD in range(4):
    val_data_names = glob.glob(f"weights//cfg_dh_05b5_loc_test2/fold{FOLD}/val*.pth")
    print(val_data_names)
    print(50*'__')
    val_data_name = val_data_names[0]

    val_data = torch.load(val_data_name, map_location=torch.device('cpu'))
    logits = val_data.pop('logits')
    del val_data['loss']
    tmpdf = pd.DataFrame({k:v.numpy() for k,v in val_data.items()})
    classes = ['x_left', 'y_left', 'x_right', 'y_right']
    tmpdf[classes] = logits.numpy()
    tmpdfls.append(tmpdf)
xydf = pd.concat(tmpdfls)
xydf.shape

xydf.to_csv('datamount/axial_xy/cfg_dh_05b5_loc_test2.csv.gz', index = False)

'''
cfg_dh_14p2_locsag_test
'''
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
fnms = sorted(glob.glob('weights/cfg_dh_14p2_locsag_test/fold*/val*'))
val_datas = [torch.load(f, map_location=torch.device('cpu')) for f in fnms]
val_datas = {k:torch.cat([v[k].detach().cpu() for v in val_datas]) for k in val_datas[0].keys()}
tstdf = pd.DataFrame({k:val_datas[k].numpy().flatten() for k in 'series_id instance_number logits_mask'.split() })
tstdf[CLASSES]= val_datas['logits']
tstdf = tstdf.sort_values('series_id  instance_number'.split()).reset_index(drop = True)
trnfdf = pd.read_csv('datamount/train_folded_v1.csv')
tstdf = pd.merge(tstdf, trnfdf[trnfdf.series_description.str.contains('Sag')]['series_id study_id'.split()], on = 'series_id', how = 'inner')

tstdf = pd.melt(tstdf,
          id_vars=['instance_number', 'logits_mask', 'series_id', 'study_id'],
          value_vars=['x__l1_l2', 'y__l1_l2', 'x__l2_l3', 'y__l2_l3', 'x__l3_l4', 'y__l3_l4', 'x__l4_l5', 'y__l4_l5', 'x__l5_s1', 'y__l5_s1'],
          var_name='level', value_name='coord')
tstdf[['coord_type', 'level']] = tstdf.level.str.split('__', expand=True)
#tstdf = tstdf.query('logits_mask > 0.2')
tstdf['coord'] = tstdf['coord'].astype(np.float32)
tstdf['logits_mask'] = tstdf['logits_mask'].astype(np.float32)
tstdf = tstdf.pivot_table(index=['instance_number', 'series_id', 'study_id', 'level', 'logits_mask'],
                          columns='coord_type',
                          values='coord').reset_index()
szdf = tstdf['instance_number  series_id    study_id'.split()].drop_duplicates()
szdf = pd.merge(szdf, metadf['instance_number  series_id img_w img_h'.split()],
               on = 'instance_number  series_id'.split(), how = 'left')

tstdf = pd.merge(tstdf, szdf, on = 'instance_number  series_id    study_id'.split(), how = 'left')
tstdf['x'] = (tstdf['x'] * tstdf['img_w']).round().astype(int)
tstdf['y'] = (tstdf['y'] * tstdf['img_h']).round().astype(int)

tstdf.to_csv('datamount/sag_xy/test__cfg_dh_14p2_locsag_test.csv.gz')


