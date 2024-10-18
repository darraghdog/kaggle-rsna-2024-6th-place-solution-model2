import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd

'''

import pandas as pd
import glob, copy, sys, importlib
import torch
from postprocess.pp_dh_04 import post_process_pipeline
from metrics.metric_dh_02 import StudyLoss
sys.path.append('configs/')

CFG_NAME = "cfg_dh_12s4c"
CFG_NAME = "cfg_dh_15b"
CFG_NAME = "cfg_dh_15c"
CFG_NAME = "cfg_dh_17a"
CFG_NAME = "cfg_dh_17f"
CFG_NAME = "cfg_dh_19a"

cfg = importlib.import_module('default_config')
importlib.reload(cfg)
cfg = importlib.import_module(CFG_NAME)
importlib.reload(cfg)
cfg = copy.copy(cfg.cfg)
cfg.pp_bwd_fwd_fill = True

val_datanms = sum([glob.glob(f'weights/{CFG_NAME}/fold{ff}/val_data_seed*.pth')[:1] for ff in range(4)], [])
print(val_datanms)
val_datas = [torch.load(f, map_location=torch.device('cpu')) for f in val_datanms]


val_data = copy.deepcopy(val_datas[0])
for k in val_datas[0]:
    val_data[k] = torch.cat([v[k] for v in val_datas])
pp_out = post_process_pipeline(cfg, val_data, None)

torch.save(pp_out, f'weights/oof____{CFG_NAME}_seed1.pth')

torch.load(f'weights/oof____{CFG_NAME}_seed1.pth')


torch.load(f'weights/oof____{CFG_NAME}.pth')['logits'][0,:2]
torch.load(f'weights/oof____{CFG_NAME}_seed0.pth')['logits'][0,:2]
torch.load(f'weights/oof____{CFG_NAME}_seed1.pth')['logits'][0,:2]

scores = []
for ii in range(4):
    for iiii in range(2):
        val_datanm = glob.glob(f'weights/{CFG_NAME}/fold{ii}/val_data_seed*.pth')[iiii]
        val_data = torch.load(val_datanm, map_location=torch.device('cpu'))
        pp_out = post_process_pipeline(cfg, val_data, None)
        study_loss = self = StudyLoss()

        loss_scs, loss_nfn, loss_ss, any_loss, \
            missed_ratio_scs, missed_ratio_nfn, missed_ratio_ss, num_studies \
                        = study_loss(pp_out['logits'],pp_out['target'])

        loss = (loss_scs + loss_nfn + loss_ss + any_loss) / 4
        score = {'score_neural_foraminal_narrowing':loss_nfn.item(),
                 'score_spinal_canal_stenosis':loss_scs.item(),
                  'score_subarticular_stenosis':loss_ss.item(),
                  'score_spinal_any_severe':any_loss.item(),
                  'missed_ratio_scs' : missed_ratio_scs.item(),
                  'missed_ratio_nfn': missed_ratio_nfn.item(),
                  'missed_ratio_ss': missed_ratio_ss.item(),
                  #'missing_studies': len(val_df.study_id.unique()) - num_studies,
                  'fold' : ii,
                  'seed': iiii,
                  'num_studies': float(num_studies)}
        scores.append(score)
scdf = pd.DataFrame(scores)

COLS = "score_neural_foraminal_narrowing  score_spinal_canal_stenosis  score_subarticular_stenosis  score_spinal_any_severe".split()
scdf.filter(regex='score_|seed').groupby('seed')[COLS].mean()

m2 = scdf.filter(regex='score_|seed').mean()


COLS = ['score_neural_foraminal_narrowing', 'score_spinal_canal_stenosis']
scdf.groupby('fold').mean(0)

scdf.filter(regex='score_|fold').groupby('fold').mean(0).mean(1)


'''


def post_process_pipeline(cfg, val_data, val_df):

    FILLVAR = -100

    actdf = pd.read_csv(cfg.train_df)
    actdf = actdf.drop(columns = 'series_description series_id fold'.split())
    actdf = actdf.set_index('study_id').fillna(FILLVAR).replace({k:t for t,k in enumerate(cfg.classes)})
    actdf = actdf.reset_index().drop_duplicates().set_index('study_id')

    pp_idx = torch.stack([val_data[k].detach().cpu() for k in "study_id level condition".split()]).permute(1,0)
    acts = val_data['target'].clone().detach().cpu()
    logits = val_data['logits'].clone().detach().cpu()

    # Convert each row to a tuple and get the unique rows (along with the inverse mapping)
    pp_idx_unq, inverse_indices = torch.unique(pp_idx[:, [0, 1, 2]], dim=0, return_inverse=True)
    logits_unq = torch.stack([logits[torch.where(inverse_indices==ii)[0]].mean(0) for ii in range(1+inverse_indices.max())])
    acts_unq = torch.stack([acts[torch.where(inverse_indices==ii)[0]][0] for ii in range(1+inverse_indices.max())])

    # Fill a tensor with all combinations of study_id level condition
    base_df = pd.DataFrame([(i, j) for i in range(5) for j in range(3)], columns=['level', 'condition'])
    base_df = pd.concat([base_df.assign(study_id=sid) for sid in pp_idx_unq[:,0].unique().numpy()])
    base_df = base_df[['study_id', 'level', 'condition']]
    dd_unq = pd.DataFrame(pp_idx_unq.numpy(), columns = "study_id level condition".split())
    dd_unq = dd_unq.reset_index() .rename(columns={'index':'l_idx'})
    base_df = pd.merge(base_df, dd_unq, on = ['study_id', 'level', 'condition'], how = 'left')
    base_df['l_idx'] = base_df['l_idx'].fillna(FILLVAR).astype(int)
    pp_idx_all = torch.from_numpy(base_df[['study_id', 'level', 'condition']].values)

    # Assign logits and actuals to a filled tensor
    l_idx = base_df.query('l_idx>-100')['l_idx'].sort_values().index
    logits_all = torch.zeros((len(base_df), *logits_unq.shape[1:])).float() + FILLVAR
    logits_all[l_idx] = logits_unq.float()
    acts_all = torch.zeros((len(base_df), *acts_unq.shape[1:])).long() + FILLVAR
    acts_all[l_idx] = acts_unq

    # Melt the tensor from long to wide
    acts_all =  torch.cat((acts_all[1::3], acts_all[0::3], acts_all[2::3]), 1)
    logits_all =  torch.cat((logits_all[1::3], logits_all[0::3], logits_all[2::3]), 1)
    pp_idx_all = pp_idx_all[1::3, [0,1]]
    n_rows = len(pp_idx_all)

    acts_all = acts_all.reshape(n_rows // 5, 5, *acts_all.shape[1:]).permute(0,2,1).reshape(n_rows // 5, -1)
    logits_all = logits_all.reshape(n_rows // 5, 5, *logits_all.shape[1:]).permute(0,2,1,3).reshape(n_rows // 5, -1, 3)
    pp_idx_all = pp_idx_all[::5, 0]

    # Remove the dummy spinal_canal_stenosis
    keep_idx = [i  for i in range(30) if i not in [5,6,7,8,9]]
    acts_all = acts_all[:,keep_idx]
    logits_all = logits_all[:,keep_idx]

    if cfg.pp_bwd_fwd_fill:
        # Realign to the full actuals and roll forward
        full_act = torch.from_numpy(actdf.loc[pp_idx_all].values)

        # Forward and backard fill
        for ii in [2,1]:
            iddx = (logits_all[:,(ii-1)::5]==FILLVAR)
            logits_all[:,(ii-1)::5][iddx] = logits_all[:,(ii)::5][iddx]
        for ii in [0,1]:
            iddx = (logits_all[:,(ii+1)::5]==FILLVAR)
            logits_all[:,(ii+1)::5][iddx] = logits_all[:,(ii)::5][iddx]
    else:
        logits_all[logits_all[:,:,0]==FILLVAR] = 0

    '''
    mlls=[]
    for ml in logits_all.permute(1,0,2):
        ml1 =  ml.clone()
        mask = ml1[:,0]==FILLVAR
        ml1[mask] = ml1[~mask].mean(0).unsqueeze(0)
        mlls.append(ml1)
    logits_all = torch.stack(mlls, 1)
    '''


    pp_out = {}
    pp_out['logits'] = logits_all.clone()
    pp_out['target'] = acts_all.clone()
    if cfg.pp_bwd_fwd_fill:
        pp_out['target'] = full_act.clone()
    pp_out['study_id'] = pp_idx_all.clone()

    return pp_out
