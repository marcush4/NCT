import pickle
import numpy as np
import pandas as pd
import pdb

import sys
#sys.path.append('/home/akumar/nse/neural_control')
sys.path.append('/home/marcush/projects/neural_control/')
sys.path.append('/home/marcush/projects/neural_control/analysis_scripts/')
sys.path.append('/home/marcush/projects/github_repos')
from loaders import load_sabes, load_peanut, reach_segment_sabes, segment_peanut, load_tsao
from scipy.ndimage import gaussian_filter1d

def filter_by_dict(df, root_key, dict_filter):

    col = df[root_key].values

    filtered_idxs = []

    for i, c in enumerate(col):
        match = True
        for key, val in dict_filter.items():
            if c[key] != val:
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
    if region in ['M1', 'S1']:
        data_path = '/mnt/Secondary/data/sabes'
    elif region == 'HPC':
        data_path = '/mnt/Secondary/data/peanut/data_dict_peanut_day14.obj'
    elif region in ['AM', 'ML']:
        data_path = '/home/marcush/Data/TsaoLabData/split/degraded'
    return data_path

def load_decoding_df(region):
    if region == 'M1':
        with open('/mnt/Secondary/data/postprocessed/sabes_m1subtrunc_dec_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)

        # Filter by start time truncation only
        filt = [idx for idx in range(df.shape[0]) 
                if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        df = df.iloc[filt]
        df = apply_df_filters(df, dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})
        # filter by start time truncation and subset selection
        # filt = [idx for idx in range(df.shape[0]) 
        #         if df.iloc[idx]['loader_args']['subset'] is not None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        # df = df.iloc[filt]
    elif region == 'S1':
        with open('/mnt/Secondary/data/postprocessed/sabes_s1subtrunc_dec_df.pkl', 'rb') as f:
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
        df = apply_df_filters(df, dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})
        session_key = 'data_file'
    elif region == 'HPC':
        with open('/home/akumar/nse/fcca_neurips/decoding.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        session_key = 'epoch'
        pdb.set_trace()

    elif region in ['ML', 'AM']:
        #dataframe_path = '/home/marcush/Data/TsaoLabData/decoding_deg_230322_214006_Jamie/decoding_deg_230322_214006_Jamie_glom.pickle'
        dataframe_path = '/clusterfs/NSDS_data/FCCA/postprocessed/tsao_decode_df.pkl'

        with open(dataframe_path, 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        df = apply_df_filters(df, loader_args={'region': region})
        session_key = 'data_file'   
        
    return df, session_key


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

def get_psth(data_path, region, session, zscore=False, full_arg_tuple=None):
    if region in ['M1', 'S1']:
        data_file = session
        dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=None, high_pass=False, region=region)
        spike_rates = dat['spike_rates']
        if zscore: spike_rates = zcore_spikes(dat, region)   

        dat_segment = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])
        
        T = 30
        t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
        valid_transitions = np.arange(t.size)[t >= T]

        # (Bin size 50 ms)
        time = 50 * np.arange(T)        
        # Store trajectories for subsequent pairwise analysis
        n = spike_rates.shape[-1]
        x = np.zeros((n, time.size))
        for j in range(n):
            x_ = np.array([spike_rates])
            x_ = np.array([spike_rates[0, dat_segment['transition_times'][idx][0]:dat_segment['transition_times'][idx][0] + T, j] 
                        for idx in valid_transitions])
            #x_ = StandardScaler().fit_transform(x_.T).T
            x_ = gaussian_filter1d(x_, sigma=2)
            x_ = np.mean(x_, axis=0)
            x[j, :] = x_
            
    elif region == 'HPC':
        epoch = session
        dat = load_peanut(data_path, epoch=epoch, boxcox=None, spike_threshold=100)
        spike_rates = dat['spike_rates']
        if zscore: spike_rates = zcore_spikes(dat, region)     

        loc_file_path = '/'.join(data_path.split('/')[:-1])
        transitions = segment_peanut(dat, loc_file=loc_file_path + '/linearization_dict_peanut_day14.obj', 
                                     epoch=epoch)

        # For now, aggregate all types of transitions together.
        transitions_all = transitions[0]
        transitions_all.extend(transitions[1])

        lens = [len(t) for t in transitions_all]
        T = 100
        n = spike_rates.shape[-1]
        time = 50 * np.arange(T)        
        # Store trajectories for subsequent pairwise analysis
        n = spike_rates.shape[-1]
        x = np.zeros((n, time.size))
        for j in range(n):
            x_ = np.array([spike_rates[trans[0]:trans[0] + T, j] 
                        for trans in transitions_all])
            #x_ = StandardScaler().fit_transform(x_.T).T
            x_ = gaussian_filter1d(x_, sigma=2)
            x_ = np.mean(x_, axis=0)
            x[j, :] = x_

    elif region in ['AM', 'ML']:

        loader_args = dict(full_arg_tuple[0])
        dat = load_tsao(**loader_args)
        spike_rates = dat['spike_rates']
        if zscore: spike_rates = zcore_spikes(dat, region)

        smooth_spikes = gaussian_filter1d(spike_rates, sigma=2, axis=1)
        x = np.mean(smooth_spikes, axis=0).squeeze().T # Averaged spike rates, shape: (numUnits, time)


    return x




def get_spikes(data_path, region, session, zscore=False, full_arg_tuple=None):

    if region in ['AM', 'ML']:

        all_loader_args = np.unique(full_arg_tuple)
        regIdx = [ ('region', region) in args for args in all_loader_args]
        loader_args = dict(all_loader_args[regIdx][0])
        dat = load_tsao(**loader_args)
        spike_rates = dat['spike_rates']
        if zscore:
            spike_rates = zcore_spikes(dat, region)


    return spike_rates


