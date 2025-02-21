#!/usr/bin/env python3
import pickle
import pandas as pd
import numpy as np
import os
import sys
import scipy 
from dca.methods_comparison import JPCA 
from tqdm import tqdm

sys.path.append('/home/marcush/projects/neural_control/')
sys.path.append('/home/marcush/projects/neural_control/analysis_scripts/')
sys.path.append('/home/marcush/projects/github_repos')

from loaders import load_tsao
from region_select import *


############################################ Arg Params (check read/write paths)
decoding_glom_path = '/home/marcush/Data/TsaoLabData/neural_control_output_new/decoding_deg_230809_140453_Alfie/decoding_deg_230809_140453_Alfie_glom.pickle'
region = 'AM' 
DIM = 21
inner_reps = 1000


paradigm_session_name = os.path.splitext(os.path.basename(decoding_glom_path))[0]
rProj_base_path = os.path.dirname(decoding_glom_path)
save_path = f'{rProj_base_path}/jpca_{region}_randomcontrol_{paradigm_session_name}.dat'
############################################


df_decode, _ = load_decoding_df(region)
df_decode = pd.DataFrame(df_decode)
data_files = np.unique(df_decode['data_file'])
jDIM = DIM - 1 if DIM % 2 != 0 else DIM    # jPCA dimension must be even

resultsd3 = []

for ii, data_file in enumerate(data_files):

    # Get info from decoding df to load the raw spike rates
    df_ = apply_df_filters(df_decode, **{'data_file':data_file})
    data_path = df_['data_path'][0]

    if region in ["ML", "AM"]:
        y = get_spikes(data_path, region, data_file, full_arg_tuple=df_['full_arg_tuple'])
    else:
        y = get_spikes(data_path, region, data_file)


    # Randomly project the spike rates and fit JPCA
    for j in tqdm(range(inner_reps)):
        V = scipy.stats.special_ortho_group.rvs(y.shape[-1], random_state=np.random.RandomState(j))
        V = V[:, 0:DIM]
        # Project data
        yproj = y @ V


        result_ = {}
        result_['data_file'] = data_file
        result_['inner_rep'] = j

        jpca = JPCA(n_components=jDIM, mean_subtract=False)
        jpca.fit(yproj)
        

        result_['jeig'] = jpca.eigen_vals_

        yprojcent = np.array([y_ - y_[0:1, :] for y_ in yproj])
        dyn_range = np.array([np.max(np.abs(y_)[:, j]) for y_ in yprojcent for j in range(jDIM)])
        result_['dyn_range'] = np.mean(dyn_range)
        resultsd3.append(result_)


        print(f"Done with jPCA fit {j+1} of {inner_reps}.")

with open(save_path, 'wb') as f:
    f.write(pickle.dumps(resultsd3))            

