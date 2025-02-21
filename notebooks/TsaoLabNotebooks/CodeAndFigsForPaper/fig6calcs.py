#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
import pdb
import scipy
import pickle
from config import PATH_DICT
from region_select_alt import *
from dca.methods_comparison import JPCA

def get_rates(T, df, data_path, region, session):
    if region == 'HPC':
        loader_args = {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}
    else:
        loader_args = df.iloc[0]['loader_args']
 
    if region in ['ML', 'AM']:
        y = get_rates_jpca(data_path, region, session, loader_args, full_arg_tuple=df['full_arg_tuple'])
    else:
        y = get_rates_jpca(data_path, region, session, loader_args)

    # Restrict to trials that match the length threshold, and standardize lengths
    y = np.array([y_[0:T] for y_ in y if len(y_) > T])
    return y

def calc_on_dimreduc(T, decoding_df, region, session_key, savepath, DIM):
    jDIM = DIM - 1 if DIM % 2 != 0 else DIM    # jPCA dimension must be even
    
    sessions = np.unique(decoding_df[session_key].values)
    data_path = get_data_path(region)
    results = []
    for ii, session in enumerate(sessions):

        y = get_rates(T, decoding_df, data_path, region, session)
        for dimreduc_method in [['LQGCA', 'FCCA'], 'PCA']:
            if region in ['AM', 'ML']:
                df_filter = {session_key:session, 'fold_idx':0, 'dim':DIM,
                             'dimreduc_method':dimreduc_method, 'loader_args':{'region':region}}
                df_ = apply_df_filters(decoding_df , **df_filter)
            else:
                df_filter = {session_key:session, 'fold_idx':0, 'dim':DIM,
                             'dimreduc_method':dimreduc_method}
                df_ = apply_df_filters(decoding_df, **df_filter)
            assert(df_.shape[0] == 1)

            V = df_.iloc[0]['coef']
            if dimreduc_method == 'PCA':
                V = V[:, 0:jDIM]        

            try:
                yproj = y @ V
            except:
                pdb.set_trace()

            result_ = {}
            result_[session_key] = session
            result_['dimreduc_method'] = dimreduc_method

            # 3 fits: Look at symmetric vs. asymmetric portions of regression onto differences
            jpca = JPCA(n_components=jDIM, mean_subtract=False)
            jpca.fit(yproj)
            
            result_['jeig'] = jpca.eigen_vals_
            yprojcent = yproj
            dyn_range = np.array([np.max(np.abs(y_)[:, j]) for y_ in yprojcent for j in range(jDIM)])
            result_['dyn_range'] = np.mean(dyn_range)

            results.append(result_)

    with open(savepath, 'wb') as f:
        f.write(pickle.dumps(results))                

def calc_on_random(T, decoding_df, region, session_key, savepath, DIM, inner_reps):
    jDIM = DIM - 1 if DIM % 2 != 0 else DIM    # jPCA dimension must be even

    results = []
    sessions = np.unique(decoding_df[session_key].values)
    data_path = get_data_path(region)
    for ii, session in enumerate(sessions):
        y = get_rates(T, decoding_df, data_path, region, session)
        # Randomly project the spike rates and fit JPCA
        for j in tqdm(range(inner_reps)):
            V = scipy.stats.special_ortho_group.rvs(y.shape[-1], random_state=np.random.RandomState(j))
            V = V[:, 0:jDIM]
            # Project data
            yproj = y @ V
            result_ = {}
            result_[session_key] = session
            result_['inner_rep'] = j

            jpca = JPCA(n_components=jDIM, mean_subtract=False)
            jpca.fit(yproj)
            result_['jeig'] = jpca.eigen_vals_

            yprojcent = np.array([y_ - y_[0:1, :] for y_ in yproj])
            dyn_range = np.array([np.max(np.abs(y_)[:, j]) for y_ in yprojcent for j in range(jDIM)])
            result_['dyn_range'] = np.mean(dyn_range)
            results.append(result_)

    with open(savepath, 'wb') as f:
        f.write(pickle.dumps(results))                

dim_dict = {
    'M1': 6,
    'M1_trialized':6,
    'S1': 6,
    'HPC': 6,
    'M1_maze': 6,
    'AM': 21,
    'ML': 21
}

T_dict = {
    'M1': 20,
    'M1_trialized': 20,
    'S1': 20,
    'HPC': 20,
    'M1_maze': 20,
    'AM': 18,
    'ML': 18
}

if __name__ == '__main__':
    regions = ['AM', 'ML']

    for region in regions:
        T = T_dict[region]
        inner_reps = 1000
        decoding_df, session_key = load_decoding_df(region, **loader_kwargs[region])
        save_path = PATH_DICT['tmp'] + '/jpca_tmp_randcontrol_%s.pkl' % region
        calc_on_random(T, decoding_df, region, session_key, save_path, dim_dict[region], inner_reps)
        save_path = PATH_DICT['tmp'] + '/jpca_tmp_%s.pkl' % region
        calc_on_dimreduc(T, decoding_df, region, session_key, save_path, dim_dict[region])