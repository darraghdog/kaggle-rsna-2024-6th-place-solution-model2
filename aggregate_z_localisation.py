import os
import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import set_pandas_display
import collections
set_pandas_display()


metadf = pd.read_pickle('datamount/meta_v1.pkl')
os.makedirs('datamount/sag_slice/', exist_ok = True)
os.makedirs('datamount/axial_lvl/', exist_ok = True)

fnms = sorted(glob.glob('weights/cfg_dh_14s10a_locsag/fold*/val*'))
fnms = [f for f in fnms if "/fold-1/" not in f]
val_datas = [torch.load(f, map_location=torch.device('cpu')) for f in fnms]
val_datas = {k:torch.cat([v[k].detach().cpu() for v in val_datas]) for k in val_datas[0].keys()}
tstdf = pd.DataFrame({k:val_datas[k].numpy().flatten() for k in 'series_id instance_number'.split() })
CLASSES2 = ['left_nfn', 'right_nfn', 'scs']
tstdf[CLASSES2]= val_datas['logits']
tstdf = tstdf.groupby('series_id instance_number'.split())[CLASSES2].mean().reset_index()
tstdf = tstdf.sort_values('series_id  instance_number'.split()).reset_index(drop = True)

tstdf.to_csv('datamount/sag_slice/test__cfg_dh_14s10a_locsag.csv.gz', index = False)


# vdf = pd.read_csv('datamount/sag_xy/original_spinenet_v01.csv.gz')
locdf = pd.read_csv('datamount/sag_xy/test__cfg_dh_14p2_locsag_test.csv.gz')
slcdf = pd.read_csv('datamount/sag_slice/test__cfg_dh_14s10a_locsag.csv.gz')

metadf  = pd.read_pickle('datamount/meta_v1.pkl')
slcdf = slcdf[slcdf.filter(like = 'nfn').max(1)>0.1].reset_index(drop = True)
slcdf['side'] = slcdf.filter(like = 'nfn').idxmax(1)
jcols = "series_id  instance_number".split()
locdf = pd.merge(locdf, slcdf[jcols + ['side']], on = jcols, how = 'inner')
fcols = "series_id  instance_number level side x y ".split()
locdf = locdf[fcols].sort_values(fcols[:3]).reset_index(drop = True)

def project_to_3d(x,y,z, df):
	d = df.iloc[z]
	H, W = d.img_h, d.img_w
	sx, sy, sz = [float(v) for v in d.ImagePositionPatient]
	o0, o1, o2, o3, o4, o5, = [float(v) for v in d.ImageOrientationPatient]
	delx, dely = d.PixelSpacing

	xx = o0 * delx * x + o3 * dely * y + sx
	yy = o1 * delx * x + o4 * dely * y + sy
	zz = o2 * delx * x + o5 * dely * y + sz
	return xx,yy,zz

def view_to_world(sagittal_t2_point, z, sagittal_t2_df, image_size):

	H = sagittal_t2_df.iloc[0].img_h
	W = sagittal_t2_df.iloc[0].img_w
	scale_x = 1.#W / image_size
	scale_y = 1.#H / image_size

	xxyyzz = []
	for l in range(1, 6):
		x,y = sagittal_t2_point.iloc[l-1]['x y'.split()]
		xx,yy,zz = project_to_3d(x*scale_x, y*scale_y, 0, sagittal_t2_df)
		xxyyzz.append((xx, yy, zz))

	xxyyzz = np.array(xxyyzz)
	return xxyyzz

'''
https://github.com/Deep-learning-exp/M-SCAN/blob/d1af1aaa35fcc4bb6ed65f6d6b5bae9120a76ea7/UNET/data.py#L207
'''

def point_to_level2(world_point, axial_t2_df):
    # we get closest axial slices (z) to the CSC world points
    xxyyzz = world_point
    orientation = np.array(axial_t2_df.ImageOrientationPatient.values.tolist())
    position = np.array(axial_t2_df.ImagePositionPatient.values.tolist())
    ox = orientation[:, :3]
    oy = orientation[:, 3:]
    oz = np.cross(ox, oy)
    t = xxyyzz.reshape(-1, 1, 3) - position.reshape(1, -1, 3)
    dis = (oz.reshape(1, -1, 3) * t).sum(-1)  # np.dot(point-s,oz)
    fdis = np.fabs(dis)
    closest_z = fdis.argmin(-1)
    closest_fdis = fdis.min(-1)
    closest_df = axial_t2_df.iloc[closest_z]
    instance_numbers = closest_df['instance_number'].tolist()
    return closest_df['instance_number'].tolist(), closest_fdis
#return assigned_level, closest_z, dis #dis is soft assignment

pcols = "PixelSpacing img_w  img_h ImageOrientationPatient ImagePositionPatient".split()
metadf = pd.merge(metadf[pcols+jcols] , slcdf[jcols], on = jcols, how = 'inner')
metadf = metadf.set_index('series_id')

xxyyzzdf = []
for (series_id, instance_number, side), g in \
    locdf.groupby("series_id  instance_number side".split()):
    mdf = metadf.loc[series_id].query("instance_number == @instance_number")
    sagittal_t2_point, sagittal_t2_df, image_size = g, mdf, (512,512)
    xxyyzz = view_to_world(sagittal_t2_point, 0, sagittal_t2_df, image_size)
    for level, xyz in zip(g.level, xxyyzz ):
        xxyyzzdf.append([series_id, instance_number, level, side, *xyz])

xxyyzzdf = pd.DataFrame(xxyyzzdf, columns = "series_id instance_number level side xx yy zz".split())
xxyyzzdf = xxyyzzdf.sort_values("series_id level instance_number".split()).reset_index(drop = True)


xxyyzzdf


metadf  = pd.read_pickle('datamount/meta_v1.pkl')\
        .sort_values('series_id instance_number'.split())
kcols = "series_id instance_number ImageOrientationPatient ImagePositionPatient PixelSpacing".split()
metadf  = metadf[kcols].set_index('series_id')

trnsdf = pd.read_csv("datamount/train_series_descriptions.csv")
mapper = collections.defaultdict(list)
for study_id, grp in trnsdf.groupby('study_id'):
    sagt1ls =  grp.query('series_description=="Sagittal T1"').series_id.tolist()
    axt2ls =  grp.query('series_description=="Axial T2"').series_id.tolist()
    for sagt1 in sagt1ls:
        mapper[sagt1] += axt2ls



axmapdf = []
for (series_id, instance_number), g in xxyyzzdf.groupby("series_id instance_number".split()):
    for axt2_series in mapper[series_id]:
        axial_t2_df = metadf.loc[axt2_series]
        world_point = g['xx yy zz'.split()].values
        ax_instance_numbers, fdis = point_to_level2(world_point, axial_t2_df)
        tmpdf = {'level': g.level.tolist(),
         'mapped_ax_instance_number': ax_instance_numbers,
         'source_series_id': series_id,
         'series_id':axt2_series,
         'distance': fdis}
        axmapdf.append(pd.DataFrame(tmpdf))


axmapdf = pd.concat(axmapdf)
axmapdf = axmapdf.sort_values('series_id level distance'.split()).reset_index(drop= True)

axmapdf = axmapdf.groupby('series_id level source_series_id'.split())\
    .agg({'mapped_ax_instance_number':list, 'distance':list})
axmapdf = axmapdf.reset_index()
#axmapdf['distance'].apply(lambda x: x[0])
axmapdf.to_pickle('datamount/axial_lvl/test__dh_14p2___dh_14s10a____locsag_mapped_v02.pkl')