import pickle
import numpy as np
import pandas as pd
import pdb
import os

import sys
from config import PATH_DICT
sys.path.append(PATH_DICT['repo'])

from loaders import (load_sabes, load_peanut, reach_segment_sabes,
                      segment_peanut, load_shenoy_large, load_tsao)
from scipy.ndimage import gaussian_filter1d

loader_kwargs = {
    'M1': {},
    'M1_trialized': {'load_idx':0},
    'S1': {},
    'HPC': {}, 
    'M1_maze': {'load_idx':8, 'dec_idx':0, 'dr_idx':3},
    'ML': {},
    'AM': {}
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
                        matching_idxs.extend(list(np.setdiff1d(np.arange(filtered_df.shape[0]), list(df_.index))))
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
    if region in ['M1', 'S1', 'M1_trialized']:
        root_path = '/home/ankit_kumar/Data'
        data_path = 'sabes'
    elif region == 'HPC':
        data_path = 'peanut/data_dict_peanut_day14.obj'
    elif region in ['M1_maze', 'M1_maze_smooth']:
        data_path = '000070'
    elif region in ['AM', 'ML']:
        data_path = 'degraded'

    return root_path + '/' + data_path

def load_decoding_df(region, **kwargs):
    root_path = PATH_DICT['df']
    if region == 'M1':
        with open(root_path + '/sabes_m1subtrunc_dec_df.pkl', 'rb') as f:
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
            load_idx = loader_kwargs['load_idx']
        else:
            load_idx = kwargs['load_idx']

        df = apply_df_filters(df, loader_args=loader_args[load_idx])
        session_key = 'data_file'

    elif region == 'S1':
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
    elif region == 'HPC':
        with open(root_path + '/hpc_decoding.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
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
        with open(root_path + dataframe_path, 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        df = apply_df_filters(df, loader_args={'region': region})
        session_key = 'data_file'   

    return df, session_key

def load_data(data_path, region, session, loader_args, full_arg_tuple=None):
    if region in ['M1', 'M1_trialized', 'S1']:
        data_file = session
        if 'region' not in loader_args:
            loader_args['region'] = region
        loader_args['high_pass'] = True
        dat = load_sabes('%s/%s' % (data_path, data_file), **loader_args)            
    elif region == 'HPC':
        epoch = session
        dat = load_peanut(data_path, epoch=epoch, **loader_args)
    elif region == 'M1_maze':
        data_file = session
        # Add the sub-folder
        subfolder = data_file.split('_ses')[0]
        dat = load_shenoy_large('%s/%s/%s' % (data_path, subfolder, data_file), **loader_args)
    elif region in ['AM', 'ML']:

        # Get correct loader args (filter by session)
        all_loader_args = np.unique(full_arg_tuple)
        sessInd = [session in ' '.join(map(str, args)) for args in all_loader_args]
        loader_args = dict(all_loader_args[sessInd][0])
        #loader_args = dict(full_arg_tuple[0])

        # Overwrite data path
        loader_args['data_path'] = data_path +'/'+ os.path.basename(loader_args['data_path'])
        dat = load_tsao(**loader_args)
    return dat

def get_rates(data_path, region, session, boxcox=False, full_arg_tuple=None):
    if boxcox:
        boxcox = 0.5
    else:
        boxcox = None

    if region in ['M1', 'S1', 'M1_trialized']:
        data_file = session
        if region == 'M1_trialized':
            region = 'M1'
        dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=boxcox, high_pass=True, region=region)
        dat_segment = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])
        T = 30
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]

        # (Bin size 50 ms)
        time = 50 * np.arange(T)        
        # Store trajectories for subsequent pairwise analysis
        n = dat['spike_rates'].shape[-1]
        x = np.zeros((n, time.size))
        for j in range(n):
            x_ = np.array([dat['spike_rates']])
            x_ = np.array([dat['spike_rates'][0, dat_segment['transition_times'][idx][0]:dat_segment['transition_times'][idx][0] + T, j] 
                        for idx in valid_transitions])
            #x_ = StandardScaler().fit_transform(x_.T).T
            x_ = gaussian_filter1d(x_, sigma=2)
            x_ = np.mean(x_, axis=0)
            x[j, :] = x_
            
    elif region == 'HPC':
        epoch = session
        dat = load_peanut(data_path, epoch=epoch, boxcox=boxcox, spike_threshold=100)
        loc_file_path = '/'.join(data_path.split('/')[:-1])
        transitions = segment_peanut(dat, loc_file=loc_file_path + '/linearization_dict_peanut_day14.obj', 
                                     epoch=epoch)

        # For now, aggregate all types of transitions together.
        transitions_all = transitions[0]
        transitions_all.extend(transitions[1])

        lens = [len(t) for t in transitions_all]
        T = 100
        n = dat['spike_rates'].shape[-1]
        time = 50 * np.arange(T)        
        # Store trajectories for subsequent pairwise analysis
        n = dat['spike_rates'].shape[-1]
        x = np.zeros((n, time.size))
        for j in range(n):
            x_ = np.array([dat['spike_rates'][trans[0]:trans[0] + T, j] 
                        for trans in transitions_all])
            #x_ = StandardScaler().fit_transform(x_.T).T
            x_ = gaussian_filter1d(x_, sigma=2)
            x_ = np.mean(x_, axis=0)
            x[j, :] = x_

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
            x_ = np.array([r[0:T, j] for idx, r in enumerate(dat['spike_rates']) if idx in valid_transitions])
            #x_ = StandardScaler().fit_transform(x_.T).T
            x_ = gaussian_filter1d(x_, sigma=2)
            x_ = np.mean(x_, axis=0)
            x[j, :] = x_

    elif region in ['AM', 'ML']:
        all_loader_args = np.unique(full_arg_tuple)
        regIdx = [ ('region', region) in args for args in all_loader_args]
        loader_args = dict(all_loader_args[regIdx][0])
        # Overwrite data path
        loader_args['data_path'] = data_path + '/' + session
        dat = load_tsao(**loader_args)
        smooth_spikes = gaussian_filter1d(dat['spike_rates'], sigma=2, axis=1)
        x = np.mean(smooth_spikes, axis=0).squeeze().T # Averaged spike rates, shape: (numUnits, time)

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

def get_rates_jpca(data_path, region, session, loader_args, full_arg_tuple=None, zscore=False):

    dat = load_data(data_path, region, session, loader_args, full_arg_tuple)
    if region in ['M1', 'S1']:
        dat_segment = reach_segment_sabes(dat, data_file=session.split('.mat')[0])
        spike_rates = [dat['spike_rates'].squeeze()[t0:t1] 
                        for t0, t1 in dat_segment['transition_times']]
    elif region in ['HPC']:
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
    return spike_rates