import pickle
import numpy as np
import pandas as pd
import pdb
import itertools
import sys
import os
from config import PATH_DICT
sys.path.append(PATH_DICT['repo'])

from loaders import (load_sabes, load_peanut, reach_segment_sabes,
                      segment_peanut, load_shenoy_large, load_tsao, trialize_franklab,
                      load_franklab_new,load_AllenVC)
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

loader_kwargs = {
    'M1': {'load_idx':0, 'dec_idx':0, 'dr_idx':0},
    'M1_psid': {'load_idx':0, 'dec_idx':0, 'dr_idx':0},
    'M1_trialized': {'load_idx':0, 'dec_idx':0, 'dr_idx':0},
    'S1': {'load_idx':0, 'dec_idx':0, 'dr_idx':0},
    'S1_psid': {'load_idx':0, 'dec_idx':0, 'dr_idx':0},
    'HPC_peanut': {'load_idx':0, 'dec_idx':0, 'dr_idx':0}, 
    'M1_maze': {'load_idx':8, 'dec_idx':0, 'dr_idx':3},
    'ML': {'load_idx':0, 'dec_idx':0, 'dr_idx':0},
    'AM': {'load_idx':0, 'dec_idx':0, 'dr_idx':0},
    'HPC': {'load_idx':18, 'dec_idx':1, 'dr_idx':1},
    'mPFC': {'load_idx':16, 'dec_idx':1, 'dr_idx':1},
    'VISp':{'load_idx':0, 'dec_idx':0, 'dr_idx':0}
}



def filter_by_dict(df, root_key, dict_filter):

    col = df[root_key].values

    filtered_idxs = []

    for i, c in enumerate(col):
        match = True
        for key, val in dict_filter.items():
            if key in c.keys():
                if c[key] != val:
                    match = False
            else:
                match = False
        if match:
            filtered_idxs.append(i)

    return df.iloc[filtered_idxs]

# Shortcut to apply multiple filters to pandas dataframe
def apply_df_filters(dtfrm, invert=False, reset_index=True, **kwargs):

    filtered_df = dtfrm

    for key, value in kwargs.items():

        # If the value is the dict
        if type(value) == dict:
            filtered_df = filter_by_dict(filtered_df, key, value)
        else:
            if type(value) == list:
                matching_idxs = []
                for v in value:
                    df_ = apply_df_filters(filtered_df, reset_index=False, **{key:v})
                    if invert:
                        matching_idxs.extend(list(np.setdifff1d(np.arange(filtered_df.shape[0]), list(df_.index))))
                    else:
                        matchings_idxs = matching_idxs.extend(list(df_.index))

                filtered_df = filtered_df.iloc[matching_idxs]
        
            elif type(value) == str:
                filtered_df = filtered_df.loc[[value in s for s in filtered_df[key].values]]
            else:
                if invert:
                    filtered_df = filtered_df.loc[filtered_df[key] != value]
                else:
                    filtered_df = filtered_df.loc[filtered_df[key] == value]
        
        if reset_index:
            filtered_df.reset_index(inplace=True, drop=True)

        # if filtered_df.shape[0] == 0:
        #     print('Key %s reduced size to 0!' % key)

    return filtered_df

def get_data_path(region):
    root_path = PATH_DICT['data']
    if region in ['M1', 'S1', 'M1_trialized', 'M1_psid', 'S1_psid']:
        # root_path = '/home/ankit_kumar/Data'
        data_path = 'sabes'
    elif region == 'HPC_peanut':
        data_path = 'peanut'
        #data_path = 'peanut/data_dict_peanut_day14.obj'
    elif region in ['M1_maze', 'M1_maze_smooth']:
        data_path = '000070'
    elif region in ['AM', 'ML']:
        data_path = 'degraded'
        #data_path = 'FOB'
    elif region in ['mPFC', 'HPC']:
        root_path = '/clusterfs/NSDS_data/franklabdata'
        data_path = 'dataset1'
    elif region in ['VISp']:
        data_path = 'AllenData'
    return root_path + '/' + data_path

def load_supervised_decoding_df(region, **kwargs):
    root_path = PATH_DICT['df']
    if region == 'M1_psid':
        with open(root_path + '/sabes_supervised_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        dim_key = 'state_dim'
    elif region == 'S1_psid':
        with open(root_path + '/sabes_supervised_dfS1.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        dim_key = 'state_dim'
    elif region == 'HPC_peanut':
        dim_key = 'ranks'
        with open(root_path + '/peanut_supervised_decoding25_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)    
        # Add epoch as a top level key and remove from loader args
        epochs = [df.iloc[k]['loader_args']['epoch'] for k in range(df.shape[0])]
        df['epoch'] = epochs
        # Pop epoch as a key in loader args to prevent double passing this 
        # argument down the line
        for k in range(df.shape[0]):
            try:
                del df.iloc[k]['loader_args']['epoch']
            except KeyError:
                pass

        filt = [idx for idx in range(df.shape[0])
                if df.iloc[idx]['decoder_args']['decoding_window'] == 12]
        df = df.iloc[filt]

    elif region in ['AM', 'ML']:
        with open(root_path + '/jamie_supervised_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        dfj = pd.DataFrame(rl)    

        with open(root_path + '/alfie_supervised_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        dfa = pd.DataFrame(rl)    

        df = pd.concat([dfj, dfa])
        df = apply_df_filters(df, loader_args={'region': region})
        dim_key = 'dummy_dim'
        # Add a dummy dimension for compatibility with plotting
        for j in range(df.shape[0]):
            df.iloc[j]['decoder_args']['dummy_dim'] = np.arange(1, 61)

    return df, dim_key


def load_rand_decoding_df(region, **kwargs):
    root_path = PATH_DICT['df']
    if region == 'M1_psid':
        with open(root_path + '/sabes_rand_decoding_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
    elif region == 'S1_psid':
        with open(root_path + '/sabes_rand_decoding_dfS1.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
    elif region == 'HPC_peanut':
        with open(root_path + '/peanut_rand_decoding25_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)    
        # Add epoch as a top level key and remove from loader args
        epochs = [df.iloc[k]['loader_args']['epoch'] for k in range(df.shape[0])]
        df['epoch'] = epochs
        # Pop epoch as a key in loader args to prevent double passing this 
        # argument down the line
        for k in range(df.shape[0]):
            try:
                del df.iloc[k]['loader_args']['epoch']
            except KeyError:
                pass
        filt = [idx for idx in range(df.shape[0])
                if df.iloc[idx]['decoder_args']['decoding_window'] == 12]
        df = df.iloc[filt]
    elif region in ['AM', 'ML']:
        with open(root_path + '/jamie_rand_decoding_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)       
        df = apply_df_filters(df, loader_args={'region': region})
    return df

def load_decoding_df(region, **kwargs):
    root_path = PATH_DICT['df']
    if region in ['M1', 'M1_psid']:
        if region == 'M1':
            with open(root_path + '/sabes_m1subtrunc_dec_df.pkl', 'rb') as f:
                rl = pickle.load(f)
        elif region == 'M1_psid':
            with open(root_path + '/sabes_psid_decoding_dfM1.pkl', 'rb') as f:
                rl = pickle.load(f)

        df = pd.DataFrame(rl)

        # Filter by start time truncation only
        filt = [idx for idx in range(df.shape[0]) 
                if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        df = df.iloc[filt]
        df_pca = apply_df_filters(df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(df, dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})
        df = pd.concat([df_pca, df_fcca])
        # filter by start time truncation and subset selection
        # filt = [idx for idx in range(df.shape[0]) 
        #         if df.iloc[idx]['loader_args']['subset'] is not None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        # df = df.iloc[filt]
        session_key = 'data_file'
    elif region == 'M1_trialized':
        with open(root_path + '/sabes_trialized_M1decoding_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)

        loader_args = [
            {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 
            'region':'M1', 'truncate_start':True},
            {'bin_width':25, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100,
             'region':'M1', 'truncate_start':True}             
        ]

        if 'load_idx' not in kwargs:
            load_idx = loader_kwargs[region]['load_idx']
        else:
            load_idx = kwargs['load_idx']

        df = apply_df_filters(df, loader_args=loader_args[load_idx])
        session_key = 'data_file'

    elif region in ['S1']:
        
        with open(root_path + '/sabes_s1subtrunc_dec_df.pkl', 'rb') as f:
            rl = pickle.load(f)

        df = pd.DataFrame(rl)

        # Filter by start time truncation only
        filt = [idx for idx in range(df.shape[0]) 
                if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        df = df.iloc[filt]


        # filter by start time truncation and subset selection
        # filt = [idx for idx in range(df.shape[0]) 
        #         if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        # df = df.iloc[filt]

        # Filter by decoder args
        filt = [idx for idx in range(df.shape[0])
                if df.iloc[idx]['decoder_args']['trainlag'] == 2]
        df = df.iloc[filt]
        df_pca = apply_df_filters(df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(df, dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})
        df = pd.concat([df_pca, df_fcca])
        session_key = 'data_file'
    elif region == 'S1_psid':
        with open(root_path + '/sabes_psid_decoding_dfS1.pkl', 'rb') as f:
            rl = pickle.load(f)

        df = pd.DataFrame(rl)
        filt = [idx for idx in range(df.shape[0]) 
                if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        df = df.iloc[filt]
        df_pca = apply_df_filters(df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(df, dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})
        df = pd.concat([df_pca, df_fcca])
        session_key = 'data_file'

    elif region == 'HPC_peanut':
        with open(root_path + '/peanut_decoding25_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        # Need to add epoch as a top level key
        epochs = [df.iloc[k]['loader_args']['epoch'] for k in range(df.shape[0])]
        df['epoch'] = epochs

        # Filter arguments
        df_pca = apply_df_filters(df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(df, dimreduc_args={'T':5, 'n_init':10, 'rng_or_seed':42})
        df = pd.concat([df_pca, df_fcca])
        filt = [idx for idx in range(df.shape[0])
                if df.iloc[idx]['decoder_args']['decoding_window'] == 12]
        df = df.iloc[filt]
        
        # Pop epoch as a key in loader args to prevent double passing this 
        # argument down the line
        for k in range(df.shape[0]):
            try:
                del df.iloc[k]['loader_args']['epoch']
            except KeyError:
                pass
        session_key = 'epoch'

    elif region == 'M1_maze':
        with open(root_path + '/mcmaze_decoding_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        data_files = np.unique(df['data_file'].values)
        # Restricting to behaviorally predictive sessions
        data_files = data_files[[0, 1, 2, 3, 8]]
        df = apply_df_filters(df, data_file=list(data_files))
        # Select from loader and decoder args
        loader_args = []
        bw = np.array([10, 25, 50])
        for b in bw:
            loader_args.append({'bin_width':b, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':1,
                                'trialize':False})
        for b in bw:
            loader_args.append({'bin_width':b, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':1,
                                'trialize':True, 'interval':'full', 'trial_threshold':0.5})
            loader_args.append({'bin_width':b, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':1,
                                'trialize':True, 'interval':'after_go', 'trial_threshold':0.5})

        dimreduc_args = [{'T':3, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42},
                         {'T':2, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42},
                         {'T':1, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42},
                         {'T':5, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42}]

        decoder_args = [{'trainlag': 3, 'testlag': 3, 'decoding_window':5},
                        {'trainlag': 1, 'testlag': 1, 'decoding_window':5},
                        {'trainlag': 2, 'testlag': 2, 'decoding_window':5}]

        if 'load_idx' not in kwargs:
            load_idx = loader_kwargs['M1_maze']['load_idx']
        else:
            load_idx = kwargs['load_idx']
        if 'dec_idx' not in kwargs:
            dec_idx = loader_kwargs['M1_maze']['dec_idx']
        else:
            dec_idx = kwargs['dec_idx']
        if 'dr_idx' not in kwargs:
            dr_idx = loader_kwargs['M1_maze']['dr_idx']
        else:
            dr_idx = kwargs['dr_idx']

        df = apply_df_filters(df, loader_args=loader_args[load_idx], decoder_args=decoder_args[dec_idx])
        # select FCCA hyperparameters and then re-merge
        df_pca = apply_df_filters(df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(df, dimreduc_method='LQGCA', dimreduc_args=dimreduc_args[dr_idx])
        df = pd.concat([df_pca, df_fcca])
        session_key = 'data_file'
    elif region == 'M1_maze_smooth':
        raise ValueError('Need to narrow down the sessions to those predictive of behavior')
        # Need to switch over to decoding df
        # Select from loader and decoder args
        loader_args = []
        bw = np.array([5, 10, 25, 50])
        sigma = np.array([0.5, 1, 2, 4, 8])
        for b in bw:
            for s in sigma:
                loader_args.append({'bin_width':b, 'filter_fn':'gaussian', 'filter_kwargs':{'sigma':s}, 'boxcox':0.5, 'spike_threshold':1,
                                    'trialize':True, 'interval':'after_go', 'trial_threshold':0.5})
                loader_args.append({'bin_width':b, 'filter_fn':'gaussian', 'filter_kwargs':{'sigma':s}, 'boxcox':0.5, 'spike_threshold':1,
                                    'trialize':False})
        dimreduc_args = [{'T':3, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42},
                         {'T':5, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42}]

        decoder_args = [
            {'trainlag': 3, 'testlag': 3, 'decoding_window':5},
            {'trainlag': 1, 'testlag': 1, 'decoding_window':5}            
        ]

        with open('/mnt/Data/neural_control/mcmaze_smooth_decoding_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)

        if 'load_idx' not in kwargs:
            load_idx = 0
        else:
            load_idx = kwargs['load_idx']

        if 'dr_idx' not in kwargs:
            dr_idx = 0
        else:
            dr_idx = kwargs['dr_idx']

        if 'dec_idx' not in kwargs:
            dec_idx = 0
        else:
            dec_idx = kwargs['dec_idx']

        df = apply_df_filters(df, loader_args=loader_args[load_idx], decoder_args=decoder_args[dec_idx])
        # select FCCA hyperparameters and then re-merge
        df_pca = apply_df_filters(df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(df, dimreduc_method='LQGCA', dimreduc_args=dimreduc_args[dr_idx])
        df = pd.concat([df_pca, df_fcca])
        session_key = 'data_file'

    elif region in ['ML', 'AM']:
        # dataframe_path = '/decoding_deg_230322_214006_Jamie_glom.pickle'
        dataframe_path = '/tsao_decode_df.pkl'
        #dataframe_path = '/tsao_decode_df_clearOnly.pkl' #/tsao_decode_df_degOnly.pkl'
        with open(root_path + dataframe_path, 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        df = apply_df_filters(df, loader_args={'region': region})
        session_key = 'data_file'   

    elif region in ['mPFC', 'HPC']:
        
        bin_width = [10, 25, 50, 100]
        trialize = [True, False]
        spike_threshold = [0, 100]
        regions_lArgs = ['mPFC', 'HPC'] 
        loader_combs = itertools.product(bin_width, trialize, spike_threshold, regions_lArgs)
        loader_args = []
        for lArgs in loader_combs:
            bWidth, bTrialize, spike_thresh, reg = lArgs
            if bTrialize:
                loader_args.append({'bin_width':bWidth, 'region':reg, 'spike_threshold':spike_thresh,  'speed_threshold':False, 'trialize':bTrialize})
            else:
                loader_args.append({'bin_width':bWidth, 'region':reg, 'spike_threshold':spike_thresh, 'speed_threshold':False, 'trialize':bTrialize})
                loader_args.append({'bin_width':bWidth, 'region':reg, 'spike_threshold':spike_thresh, 'speed_threshold':True, 'trialize':bTrialize})

        dimreduc_args = []
        for T in np.array([1, 3, 5]):
            dimreduc_args.append({'T':T, 'loss_type':'trace', 'n_init':10})

        decoders = [{'trainlag': 0, 'testlag': 0, 'decoding_window': 6},
                    {'trainlag': 1, 'testlag': 1, 'decoding_window': 6},
                    {'trainlag': -1, 'testlag': -1, 'decoding_window': 6},
                    {'trainlag': 2, 'testlag': 2, 'decoding_window': 6},
                    {'trainlag': -2, 'testlag': -2, 'decoding_window': 4}]
        
        if 'load_idx' not in kwargs:
            load_idx = 0
        else:
            load_idx = kwargs['load_idx']

        if 'dr_idx' not in kwargs:
            dr_idx = 0
        else:
            dr_idx = kwargs['dr_idx']

        if 'dec_idx' not in kwargs:
            dec_idx = 0
        else:
            dec_idx = kwargs['dec_idx']

        #dataframe_path = '/mPFC_decoding_df.pkl'
        dataframe_path = '/decoding_fullarg_frank_lab_glom.pickle'

        with open(root_path + dataframe_path, 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        df = apply_df_filters(df, loader_args=loader_args[load_idx], decoder_args=decoders[dec_idx])
        df_pca = apply_df_filters(df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(df, dimreduc_method='LQGCA', dimreduc_args=dimreduc_args[dr_idx])
        df = pd.concat([df_pca, df_fcca])

        session_key = 'data_file'

    elif region == 'VISp':
        # Get entire DF
        with open(root_path + '/decoding_AllenVC_VISp_glom.pickle', 'rb') as f:            
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        session_key = 'data_file'
                
        # Filter it for certain load params (if applicable)
        if 'load_idx' not in kwargs:
            load_idx = loader_kwargs[region]['load_idx']
        else:
            load_idx = kwargs['load_idx']
            
        unique_loader_args = list({frozenset(d.items()) for d in df['loader_args']})
        df = apply_df_filters(df, loader_args=dict(unique_loader_args[load_idx]))
        
    return df, session_key

def load_data(data_path, region, session, loader_args, full_arg_tuple=None):
    if region in ['M1', 'M1_trialized', 'S1', 'M1_psid', 'S1_psid']:
        data_file = session
        if 'region' not in loader_args:
            loader_args['region'] = region
        loader_args['high_pass'] = True
        dat = load_sabes('%s/%s' % (data_path, data_file), **loader_args)            
    elif region == 'HPC_peanut':
        epoch = session
        dat = load_peanut(data_path, epoch=epoch, **loader_args)
    elif region == 'M1_maze':
        data_file = session
        # Add the sub-folder
        subfolder = data_file.split('_ses')[0]
        dat = load_shenoy_large('%s/%s/%s' % (data_path, subfolder, data_file), **loader_args)
    elif region in ['AM', 'ML']:
        loader_args = dict(full_arg_tuple[0])
        # Overwrite data path
        loader_args['data_path'] = data_path + '/' + os.path.basename(loader_args['data_path'])
        dat = load_tsao(**loader_args)
    elif region in ['mPFC', 'HPC']:
        dat = load_franklab_new(data_path, session=session, **loader_args)
        
    elif region in ['VISp']:        
        sess_folder = session.split(".")[0]
        path_to_data = data_path + '/' + sess_folder + "/" + session
                
        dat = load_AllenVC(path_to_data, **loader_args)
    return dat

def get_rates_smoothed(data_path, region, session, trial_average=True,
                       std=False, boxcox=False, full_arg_tuple=None, 
                       loader_args=None, return_t=False, sigma=2):
    if boxcox:
        boxcox = 0.5
    else:
        boxcox = None

    if region in ['M1', 'S1', 'M1_trialized']:
        data_file = session
        if region == 'M1_trialized':
            dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=boxcox, high_pass=False, region='M1')
        else:
            dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=boxcox, high_pass=False, region=region)
        dat_segment = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])
        T = 30
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]

        # (Bin size 50 ms)
        time = 50 * np.arange(T)        
        # Store trajectories for subsequent pairwise analysis
        n = dat['spike_rates'].shape[-1]

        if trial_average:
            x = np.zeros((n, time.size))
        else:
            x = np.zeros((n,), dtype=object)
        for j in range(n):
            x_ = np.array([dat['spike_rates'][0, dat_segment['transition_times'][idx][0]:dat_segment['transition_times'][idx][0] + T, j] 
                        for idx in valid_transitions])
            if std:
                x_ = StandardScaler().fit_transform(x_.T).T
            x_ = gaussian_filter1d(x_, sigma=sigma)
            if trial_average:
                x_ = np.mean(x_, axis=0)
            x[j] = x_               

    elif region == 'HPC_peanut':
        epoch = session 
        data_path = data_path+"/data_dict_peanut_day14.obj"  # MIH EDIT: ???? THIS NEEDS TO BE PATH TO FILE NOT DIR. WHY DOES DATA_PATH POINT TO DIR?     
        dat = load_peanut(data_path, epoch=epoch, boxcox=boxcox, spike_threshold=100, bin_width=25)
        loc_file_path = '/'.join(data_path.split('/')[:-1])
        transitions = segment_peanut(dat, loc_file=loc_file_path + '/linearization_dict_peanut_day14.obj', 
                                     epoch=epoch)

        # For now, aggregate all types of transitions together.
        transitions_all = transitions[0]
        transitions_all.extend(transitions[1])

        lens = [len(t) for t in transitions_all]
        T = 100
        n = dat['spike_rates'].shape[-1]
        time = 25 * np.arange(T)        
        # Store trajectories for subsequent pairwise analysis
        n = dat['spike_rates'].shape[-1]
        if trial_average:
            x = np.zeros((n, time.size))
        else:
            x = np.zeros((n,), dtype=object)
        for j in range(n):
            # Are all trials longer than T?
            assert(np.all([len(t) > T for t in transitions_all]))
            x_ = np.array([dat['spike_rates'][trans[0]:trans[0] + T, j] 
                        for trans in transitions_all])

            if std:
                x_ = StandardScaler().fit_transform(x_.T).T
            x_ = gaussian_filter1d(x_, sigma=sigma)
            if trial_average:
                x_ = np.mean(x_, axis=0)
            x[j] = x_
    elif region == 'M1_maze':
        data_file = session
        # Add the sub-folder
        subfolder = data_file.split('_ses')[0]
        bin_width = 50
        dat = load_shenoy_large('%s/%s/%s' % (data_path, subfolder, data_file), bin_width=bin_width,
                                boxcox=boxcox,
                                trialize=True, interval='after_go', trial_threshold=0.5, spike_threshold=1)
 

        # The cutoff should be 2000 ms. Adjust according to bin size
        T = int(2000/bin_width)
        t = np.array([x.shape[0] for x in dat['spike_rates']])
        valid_transitions = np.arange(t.size)[t >= T]

        # (Bin size 50 ms)
        time = bin_width * np.arange(T)        
        # Store trajectories for subsequent pairwise analysis
        n = dat['spike_rates'][0].shape[-1]
        x = np.zeros((n, time.size))
        for j in range(n):
            x_ = np.array([r[0:T, j] for idx, r in enumerate(dat['spike_rates']) 
                           if idx in valid_transitions])
            if std: 
                x_ = StandardScaler().fit_transform(x_.T).T
            x_ = gaussian_filter1d(x_, sigma=sigma)
            x_ = np.mean(x_, axis=0)
            x[j, :] = x_

    elif region in ['AM', 'ML']:
        
        loader_args = dict(np.unique(full_arg_tuple)[int('Jamie' not in session)])
        # Overwrite data path
        loader_args['data_path'] = data_path + '/' + session
        dat = load_tsao(**loader_args)
        
        x = dat['spike_rates']
        x = x[:,:int(x.shape[1]/2.2),:] #cut trrials in half, sine second half is gray screen
        x = gaussian_filter1d(x, sigma=1.2, axis=1)

        # if zscore: spike_rates = zcore_spikes(dat, region)
        if std:
            y = StandardScaler().fit_transform(x.reshape(-1, x.shape[-1]))
            x = y.reshape(x.shape)
        
        if trial_average:
            x = np.mean(x, axis=0).squeeze().T # Averaged spike rates, shape: (numUnits, time)
        else:
            x = x.transpose((2, 0, 1)) # reshape to n_neurons, n_trials, n_time

        T = x.shape[-1]
        time = loader_args['bin_width'] * np.arange(T)


    elif region in ["mPFC", "OFC", "HPC"]: 
        # Overwrite boxcox
        loader_args['boxcox'] = boxcox
        # Trialize
        loader_args['trialize'] = True
        # Do not speed threshold
        loader_args['speed_threshold'] = False

        # Session key / other loader args?
        dat = load_franklab_new(data_path, **loader_args)
        trial_times = [s.shape[0] for s in dat['spike_rates']]
        min_trial_time = min(trial_times)
        numUnits = dat['spike_rates'][0].shape[-1]
        x = np.zeros((numUnits, min_trial_time))

        # need to standardize to a constant trial length

        for unit in range(numUnits):
            x_ = np.array([s[0:min_trial_time, unit] for s in dat['spike_rates']])
            x_ = gaussian_filter1d(x_, sigma=sigma)
            x_ = np.mean(x_, axis=0)
            x[unit, :] = x_
            
        time = loader_args['bin_width'] * np.arange(min_trial_time)

    elif region in ['VISp']:
        
        sess_folder = session.split(".")[0]
        path_to_data = data_path + '/' + sess_folder + "/" + session
        
        dat = load_AllenVC(path_to_data, **loader_args)
        
        x = dat['spike_rates']
        x = gaussian_filter1d(x, sigma=sigma, axis=1)

        # if zscore: spike_rates = zcore_spikes(dat, region)
        if std:
            y = StandardScaler().fit_transform(x.reshape(-1, x.shape[-1]))
            x = y.reshape(x.shape)
        
        if trial_average:
            x = np.mean(x, axis=0).squeeze().T # Averaged spike rates, shape: (numUnits, time)
        else:
            x = x.transpose((2, 0, 1)) # reshape to n_neurons, n_trials, n_time
        
        
        T = x.shape[-1]
        time = loader_args['bin_width'] * np.arange(T)
        
        
    if return_t:
        return x, time
    else:   
        return x

def zcore_spikes(dat, region):

    if region in ['AM', 'ML']:
        spike_rates = dat['spike_rates']
        bin_width = dat['bin_width']

        # Take responses after this much trial time
        median_stim_on_time = 510
        window = median_stim_on_time + 120 
        threshold_bin = window // bin_width  

        baseline_responses = []
        for unit in range(spike_rates.shape[2]): 
            unit_baseline = []  
            for trial in range(spike_rates.shape[0]):
                trial_spike_rate = spike_rates[trial, :, unit]
                    
                if len(trial_spike_rate) > threshold_bin:  # Only consider trials that are long enough
                    unit_baseline.append(trial_spike_rate[threshold_bin:])
            baseline_responses.append(np.concatenate(unit_baseline))

        baseline_responses = np.vstack(baseline_responses)
        base_means = np.mean(baseline_responses, axis=1)
        base_stds = np.std(baseline_responses, axis=1)
        zscored_rates = (spike_rates - base_means) / base_stds

    else: 
        print("This dataset can not be z-scored. Returning raw spike rates.")
        zscored_rates = dat['spike_rates']

    return zscored_rates


def get_tsao_raw_rates(data_path, region, session, loader_args=None, full_arg_tuple=None):
    dat = load_data(data_path, region, session, loader_args, full_arg_tuple)
    
    maxDur = np.max(dat['TrialDurations']) + dat['bin_width']
    maxInds = int(maxDur // dat['bin_width'])
    new_rates = np.zeros((dat['NumTrials'], maxInds, dat['NumUnits']))
    
    for trialInd in np.arange(dat['NumTrials']):
        for unitInd in np.arange(dat['NumUnits']):
            
            spike_times = dat['spike_times'][trialInd, unitInd]
            trialDur = dat['TrialDurations'][trialInd]        
            bin_edges = np.arange(0, maxDur, dat['bin_width'])
            spike_counts, _ = np.histogram(spike_times, bins=bin_edges)
            new_rates[trialInd, :, unitInd] = spike_counts

    trial_inds = np.argsort(dat['StimIDs'])
    
    return new_rates, dat['spike_times'][trial_inds, :]
    
    

def get_rates_raw(data_path, region, session, loader_args=None, full_arg_tuple=None, zscore=False):
    
    dat = load_data(data_path, region, session, loader_args, full_arg_tuple)
    
    if region in ['M1', 'S1']:
        dat_segment = reach_segment_sabes(dat, data_file=session.split('.mat')[0])
        spike_rates = [dat['spike_rates'].squeeze()[t0:t1] 
                        for t0, t1 in dat_segment['transition_times']]
    elif region in ['HPC_peanut']:
        loc_file_path = '/'.join(data_path.split('/')[:-1])
        transitions = segment_peanut(dat, loc_file=loc_file_path + '/linearization_dict_peanut_day14.obj', 
                                     epoch=session)

        # For now, aggregate all types of transitions together.
        transitions_all = transitions[0]
        transitions_all.extend(transitions[1])
        spike_rates = [dat['spike_rates'][trans[0]:trans[-1], :] for trans in transitions_all]
    elif region in ['M1_maze']:
        print('Assuming trialized!')
        spike_rates = list(dat['spike_rates'])
    elif region in ['AM', 'ML']:
        spike_rates = dat['spike_rates']
        if zscore:
            spike_rates = zcore_spikes(dat, region)

    elif region in ['mPFC', 'HPC']:
        
        #spike_rates = dat['spike_rates']
        # Can instead trialize and then return spike_rates
        loader_args['trialize'] = True
        dat = load_data(data_path, region, session, loader_args, full_arg_tuple)
        spike_rates = dat['spike_rates']
        
    elif region in ['VISp']:

        spike_rates = dat['spike_rates']
        
        
    return spike_rates


def get_franklab_args(load_idx, dec_idx, dr_idx):
    
    bin_width = [10, 25, 50, 100]
    trialize = [True, False]
    spike_threshold = [0, 100]
    regions_lArgs = ['mPFC', 'HPC'] 
    loader_combs = itertools.product(bin_width, trialize, spike_threshold, regions_lArgs)
    loader_args = []
    for lArgs in loader_combs:
        bWidth, bTrialize, spike_thresh, reg = lArgs
        if bTrialize:
            loader_args.append({'bin_width':bWidth, 'region':reg, 'spike_threshold':spike_thresh,  'speed_threshold':False, 'trialize':bTrialize})
        else:
            loader_args.append({'bin_width':bWidth, 'region':reg, 'spike_threshold':spike_thresh, 'speed_threshold':False, 'trialize':bTrialize})
            loader_args.append({'bin_width':bWidth, 'region':reg, 'spike_threshold':spike_thresh, 'speed_threshold':True, 'trialize':bTrialize})

    dimreduc_args = []
    for T in np.array([1, 3, 5]):
        dimreduc_args.append({'T':T, 'loss_type':'trace', 'n_init':10})

    decoders = [{'trainlag': 0, 'testlag': 0, 'decoding_window': 6},
                {'trainlag': 1, 'testlag': 1, 'decoding_window': 6},
                {'trainlag': -1, 'testlag': -1, 'decoding_window': 6},
                {'trainlag': 2, 'testlag': 2, 'decoding_window': 6},
                {'trainlag': -2, 'testlag': -2, 'decoding_window': 4}]

    return loader_args[load_idx], decoders[dec_idx], dimreduc_args[dr_idx]

def make_hashable(d):
    """Recursively convert unhashable elements (dicts/lists) into hashable types."""
    if isinstance(d, dict):
        return frozenset((k, make_hashable(v)) for k, v in d.items())  # Convert dict to frozenset
    elif isinstance(d, list):
        return tuple(make_hashable(v) for v in d)  # Convert list to tuple
    elif isinstance(d, np.ndarray):  
        return tuple(d.tolist())  # Convert NumPy array to tuple
    else:
        return d  # Keep other types unchanged


if __name__ == "__main__":
    load_decoding_df('mPFC', **{'load_idx':16, 'dec_idx':3, 'dr_idx':2})
