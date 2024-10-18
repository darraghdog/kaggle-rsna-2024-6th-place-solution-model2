import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

'''


val_datanm = 'weights/cfg_dh_12s1/fold0/val_data_seed99310.pth'
val_data = torch.load(val_datanm, map_location=torch.device('cpu'))

import glob, copy
val_datanms = glob.glob('weights/cfg_dh_12s1/fold*/val_data_seed*.pth')
val_datas = [torch.load(f, map_location=torch.device('cpu')) for f in val_datanms]

val_data = copy.deepcopy(val_datas[0])
for k in val_datas[0]:
    val_data[k] = torch.cat([v[k] for v in val_datas])

'''


def post_process_pipeline(cfg, val_data, val_df):

    pp_idx = torch.stack([val_data[k].detach().cpu() for k in "study_id level condition".split()]).permute(1,0)
    acts = val_data['target'].clone().detach().cpu()
    logits = val_data['logits'].clone().detach().cpu()

    # Convert each row to a tuple and get the unique rows (along with the inverse mapping)
    pp_idx_unq, inverse_indices = torch.unique(pp_idx[:, [0, 1, 2]], dim=0, return_inverse=True)
    logits_unq = torch.stack([logits[torch.where(inverse_indices==ii)[0]].mean(0) for ii in range(1+inverse_indices.max())])
    acts_unq = torch.stack([acts[torch.where(inverse_indices==ii)[0]][0] for ii in range(1+inverse_indices.max())])

    # Fill a tensor with all combinations of study_id level condition
    base_df = pd.DataFrame([(i, j) for i in range(5) for j in range(2)], columns=['level', 'condition'])
    base_df = pd.concat([base_df.assign(study_id=sid) for sid in pp_idx_unq[:,0].unique().numpy()])
    base_df = base_df[['study_id', 'level', 'condition']]
    dd_unq = pd.DataFrame(pp_idx_unq.numpy(), columns = "study_id level condition".split())
    dd_unq = dd_unq.reset_index() .rename(columns={'index':'l_idx'})
    base_df = pd.merge(base_df, dd_unq, on = ['study_id', 'level', 'condition'], how = 'left')
    base_df['l_idx'] = base_df['l_idx'].fillna(-100).astype(int)
    pp_idx_all = torch.from_numpy(base_df[['study_id', 'level', 'condition']].values)

    # Assign logits and actuals to a filled tensor
    l_idx = base_df.query('l_idx>-100')['l_idx'].sort_values().index
    logits_all = torch.zeros((len(base_df), *logits_unq.shape[1:])).float()
    logits_all[l_idx] = logits_unq.float()
    acts_all = torch.zeros((len(base_df), *acts_unq.shape[1:])).long() - 100
    acts_all[l_idx] = acts_unq

    # Melt the tensor from long to wide
    acts_all =  torch.cat((acts_all[1::2], acts_all[0::2]), 1)
    logits_all =  torch.cat((logits_all[1::2], logits_all[0::2]), 1)
    pp_idx_all = pp_idx_all[1::2, [0,1]]
    n_rows = len(pp_idx_all)

    acts_all = acts_all.reshape(n_rows // 5, 5, *acts_all.shape[1:]).permute(0,2,1).reshape(n_rows // 5, -1)
    logits_all = logits_all.reshape(n_rows // 5, 5, *logits_all.shape[1:]).permute(0,2,1,3).reshape(n_rows // 5, -1, 3)
    pp_idx_all = pp_idx_all[::5, 0]

    # Remove the dummy spinal_canal_stenosis
    keep_idx = [i  for i in range(20) if i not in [5,6,7,8,9]]
    acts_all = acts_all[:,keep_idx]
    logits_all = logits_all[:,keep_idx]

    # Add on the ss columns
    acts_out = torch.zeros((len(acts_all), 25)).long()-100
    logits_out = torch.zeros((len(logits_all), 25, 3))
    acts_out[:,:15] = acts_all
    logits_out[:,:15] = logits_all

    pp_out = {}
    pp_out['logits'] = logits_out
    pp_out['target'] = acts_out
    pp_out['study_id'] = pp_idx_all

    return pp_out
