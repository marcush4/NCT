import os
import pickle
import itertools
import operator
import numpy as np
import h5py
import glob
import inspect
import warnings

# Avoid random OS errors
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
from tqdm import tqdm
from scipy import io
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from scipy.signal import resample,  convolve, get_window
from scipy.ndimage import convolve1d, gaussian_filter1d
from copy import deepcopy
import pdb
from joblib import Parallel, delayed

from segmentation import reach_segment_sabes
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

start_times = {'indy_20160426_01': 0,
               'indy_20160622_01':1700,
               'indy_20160624_03': 500,
               'indy_20160627_01': 0,
               'indy_20160630_01': 0,
               'indy_20160915_01': 0,
               'indy_20160921_01': 0,
               'indy_20160930_02': 0,
               'indy_20160930_05': 300,
               'indy_20161005_06': 0,
               'indy_20161006_02': 350,
               'indy_20161007_02': 950,
               'indy_20161011_03': 0,
               'indy_20161013_03': 0,
               'indy_20161014_04': 0,
               'indy_20161017_02': 0,
               'indy_20161024_03': 0,
               'indy_20161025_04': 0,
               'indy_20161026_03': 0,
               'indy_20161027_03': 500,
               'indy_20161206_02': 5500,
               'indy_20161207_02': 0,
               'indy_20161212_02': 0,
               'indy_20161220_02': 0,
               'indy_20170123_02': 0,
               'indy_20170124_01': 0,
               'indy_20170127_03': 0,
               'indy_20170131_02': 0,
               'loco_20170210_03':0, 
               'loco_20170213_02':0, 
               'loco_20170214_02':0, 
               'loco_20170215_02':0, 
               'loco_20170216_02': 0, 
               'loco_20170217_02': 0, 
               'loco_20170227_04': 0, 
               'loco_20170228_02': 0, 
               'loco_20170301_05':0, 
               'loco_20170302_02':0}

from pynwb import NWBHDF5IO

def filter_window(signal, window_name,  window_length=10):
    window = get_window(window_name, window_length)
    signal = convolve1d(signal, window)
    return signal

FILTER_DICT = {'gaussian':gaussian_filter1d, 'none': lambda x, **kwargs: x, 'window': filter_window}

def moving_center(X, n, axis=0):
    if n % 2 == 0:
        n += 1
    w = -np.ones(n) / n
    w[n // 2] += 1
    X_ctd = convolve1d(X, w, axis=axis)
    return X_ctd

def sinc_filter(X, fc, axis=0):
        
    # Windowed sinc filter
    b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)
    
    # Compute sinc filter.
    h = np.sinc(2 * fc * (n - (N - 1) / 2))

    # Compute Blackman window.
    w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
        0.08 * np.cos(4 * np.pi * n / (N - 1))

    # Multiply sinc filter by window.
    h = h * w

    # Normalize to get unity gain.
    h = h / np.sum(h)
    return convolve(X, h)        

def window_spike_array(spike_times, tstart, tend):
    windowed_spike_times = np.zeros(spike_times.shape, dtype=np.object)

    for i in range(spike_times.shape[0]):
        for j in range(spike_times.shape[1]):
            wst, _ = window_spikes(spike_times[i, j], tstart[i], tend[i])
            windowed_spike_times[i, j] = wst

    return windowed_spike_times

def window_spikes(spike_times, tstart, tend, start_idx=0):

    spike_times = spike_times[start_idx:]
    spike_times[spike_times > tstart]

    if len(spike_times) > 0:
        start_idx = np.argmax(spike_times > tstart)
        end_idx = np.argmin(spike_times < tend)

        windowed_spike_times = spike_times[start_idx:end_idx]

        # Offset spike_times to start at 0
        if windowed_spike_times.size > 0:
                windowed_spike_times -= tstart

        return windowed_spike_times, end_idx - 1
    else:
        return np.array([]), start_idx

def align_behavior(x, T, bin_width):
    
    bins = np.linspace(0, T, int(T//bin_width))
    bin_centers = bins + (bins[1] - bins[0])/2
    bin_centers = bin_centers[:-1]
    xaligned = np.zeros((bin_centers.size, x.shape[-1]))
    
    for j in range(x.shape[-1]):
        interpolator = interp1d(np.linspace(0, T, x[:, j].size), x[:, j])
        xaligned[:, j] = interpolator(bin_centers)

    return xaligned

def align_peanut_behavior(t, x, bins):
    # Offset to 0
    t -= t[0]
    bin_centers = bins + (bins[1] - bins[0])/2
    bin_centers = bin_centers[:-1]
    interpolator = interp1d(t, x, axis=0)
    xaligned = interpolator(bin_centers)
    return xaligned, bin_centers

# spike_times: (n_trial, n_neurons)
#  trial threshold: If we require a spike threshold, trial threshold = 1 requires 
#  the spike threshold to hold for the neuron for all trials. 0 would mean no trials

# Need to (1) speed this guy up, (2) make sure filtering is doing the right thing
# (3) remove parisitic memory usage

def postprocess_spikes(spike_times, T, bin_width, boxcox, filter_fn, filter_kwargs,
                       spike_threshold=0, trial_threshold=1, high_pass=False, return_unit_filter=False):

    # Trials are of different duration
    if np.isscalar(T):
        ragged_trials = False
    else:
        ragged_trials = True

    # Discretize time over bins
    if ragged_trials:
        bins = []
        for i in range(len(T)):
            bins.append(np.linspace(0, T[i], int(T[i]//bin_width)))
        bins = np.array(bins, dtype=np.object)
        spike_rates = np.zeros((spike_times.shape[0], spike_times.shape[1]), dtype=np.object)
    else:
        bins = np.linspace(0, T, int(T//bin_width))
        spike_rates = np.zeros((spike_times.shape[0], spike_times.shape[1], bins.size - 1,))    

    # Did the trial/unit have enough spikes?
    insufficient_spikes = np.zeros(spike_times.shape)
    #print('Processing spikes')
    #for i in tqdm(range(spike_times.shape[0])):
    for i in range(spike_times.shape[0]):
        for j in range(spike_times.shape[1]):    

            # Ignore this trial/unit combo
            if np.any(np.isnan(spike_times[i, j])):
                insufficient_spikes[i, j] = 1          

            if ragged_trials:
                spike_counts = np.histogram(spike_times[i, j], bins=np.squeeze(bins[i]))[0]    
            else:
                spike_counts = np.histogram(spike_times[i, j], bins=bins)[0]

            if spike_threshold is not None:
                if np.sum(spike_counts) <= spike_threshold:
                    insufficient_spikes[i, j] = 1

            # Apply a boxcox transformation
            if boxcox is not None:
                spike_counts = np.array([(np.power(spike_count, boxcox) - 1)/boxcox 
                                         for spike_count in spike_counts])

            # Filter only if we have to, otherwise vectorize the process
            if ragged_trials:
                # Filter the resulting spike counts
                spike_rates_ = FILTER_DICT[filter_fn](spike_counts.astype(np.float), **filter_kwargs)
                # High pass to remove long term trends (needed for sabes data)
                if high_pass:
                    spike_rates_ = moving_center(spike_rates_, 600)
            else:
                spike_rates_ = spike_counts
            spike_rates[i, j] = spike_rates_

    # Filter out bad units
    sufficient_spikes = np.arange(spike_times.shape[1])[np.sum(insufficient_spikes, axis=0) < \
                                                        (1 - (trial_threshold -1e-3)) * spike_times.shape[0]]
    spike_rates = spike_rates[:, list(sufficient_spikes)]

    # Transpose so time is along the the second 'axis'
    if ragged_trials:
        spike_rates = [np.array([spike_rates[i, j] for j in range(spike_rates.shape[1])]).T for i in range(spike_rates.shape[0])]
    else:
        # Filter the resulting spike counts
        print('FILTERING SPIKE RATES!')
        spike_rates = FILTER_DICT[filter_fn](spike_rates, **filter_kwargs)
        # High pass to remove long term trends (needed for sabes data)
        if high_pass:
            spike_rates = moving_center(spike_rates, 600, axis=-1)

        spike_rates = np.transpose(spike_rates, (0, 2, 1))

    if return_unit_filter:
        return spike_rates, sufficient_spikes
    else:
        return spike_rates

def load_cv(file_path, zscore=False):

    def zscore(data, axis=1, window=10):
        base = data[:,list(range(window)) + list(range(data.shape[axis]-window, data.shape[axis])), :]
        means = base.mean(axis=axis, keepdims=True)
        stds = base.std(axis=axis, keepdims=True)
        zdata = (data - means) / stds
        return zdata

    f = h5py.File(file_path, 'r')
    X = np.squeeze(f['X'])
    y = np.array(f['y'])

    # Remove 'thee'
    theeless = np.where(y != b'thee')[0]
    X = X[theeless, ...]
    y = y[theeless]    

    dat = {}
    dat['spike_rates'] = X
    dat['behavior'] = y

    return dat

# Loader that operates on the files provided by the Shenoy lab
def load_shenoy(data_path, bin_width, boxcox, filter_fn, filter_kwargs, 
                spike_threshold=None, trial_threshold=0.5, tw=(-250, 550), 
                trialVersions='all', trialTypes='all', region='both'):

    # Code checks for list membership in specified trialtypes/trialversions
    if trialVersions != 'all' and type(trialVersions) != list:
        trialVersions = [trialVersions]
    if trialTypes != 'all' and type(trialTypes) != list:
        trialTypes = [trialTypes]

    dat = {}
    f = io.loadmat(data_path, squeeze_me=True, struct_as_record=False)
     
    # Filter out trials we should not use, period.
    trial_filters = {'success': 0, 'possibleRTproblem' : 1, 'unhittable' : 1, 'trialType': 0, 
                     'novelMaze': 1}
    
    bad_trials = []
    for i in range(f['R'].size):
        for key, value in trial_filters.items():
            if getattr(f['R'][i], key) == value:
                bad_trials.append(i)
    bad_trials = np.unique(bad_trials)
    print('%d Bad Trials being thrown away' % bad_trials.size)
    valid_trials = np.setdiff1d(np.arange(f['R'].size), bad_trials)

    # Filter out trialVersions and trialTypes not compliant
    trialVersion = np.array([f['R'][idx].trialVersion for idx in valid_trials])
    trialType = np.array([f['R'][idx].trialType for idx in valid_trials])

    if trialVersions != 'all':
        valid_trial_versions = set([idx for ii, idx in enumerate(valid_trials)
                                    if trialVersion[ii] in trialVersions])
    else:
        valid_trial_versions = set(valid_trials)
    if trialTypes != 'all':
        valid_trial_types = set([idx for ii, idx in enumerate(valid_trials)
                                 if trialType[ii] in trialTypes])
    else:
        valid_trial_types = set(valid_trials)

    valid_trials = np.array(list(set(valid_trials).intersection(valid_trial_versions).intersection(valid_trial_types)))    
    print('%d Trials selected' % valid_trials.size)

    # Timing information
    reveal_times = np.array([f['R'][idx].actualFlyAppears for idx in valid_trials])
    go_times = np.array([f['R'][idx].actualLandingTime for idx in valid_trials])
    reach_times = np.array([f['R'][idx].offlineMoveOnsetTime for idx in valid_trials])
    total_times = np.array([f['R'][idx].HAND.X.size for idx in valid_trials])

    # Neural data - filter by requested brain region
    n_units = f['R'][0].unit.size
    spike_times = []
    unit_lookup = ['PMD' if lookup == 1 else 'M1' for lookup in f['SU'].arrayLookup]
    for i in range(valid_trials.size):
        spike_times.append([])
        for j in range(len(unit_lookup)):
            if region == 'both' or unit_lookup[j] == region:
                if np.isscalar(f['R'][i].unit[j].spikeTimes):            
                    spike_times[i].append(np.array([f['R'][i].unit[j].spikeTimes]))
                else:
                    spike_times[i].append(np.array(f['R'][i].unit[j].spikeTimes))
    
    dat['spike_times'] = np.array(spike_times).astype(np.object)
    dat['reach_times'] = reach_times


    T  = tw[1] - tw[0]
    spike_rates = postprocess_spikes(window_spike_array(dat['spike_times'], dat['reach_times'] + tw[0], 
                                                        reach_times + tw[1]),
                                                        T, bin_width, boxcox, filter_fn, filter_kwargs, 
                                                        spike_threshold=spike_threshold, 
                                                        trial_threshold=trial_threshold)                      

    dat['spike_rates'] = spike_rates         


    #### Behavior ####
    handX = np.zeros(valid_trials.size).astype(np.object)
    handY = np.zeros(valid_trials.size).astype(np.object)

    for i in range(valid_trials.size):
        handX[i] = f['R'][i].HAND.X
        handY[i] = f['R'][i].HAND.Y


    dat['behavior'] = np.zeros((spike_rates.shape[0], 
                                spike_rates.shape[1], 2))

    for i in range(spike_rates.shape[0]):
        # Align behavioral variables to binned neural data
        hand = np.vstack([handX[i][dat['reach_times'][i] - int(T/2):dat['reach_times'][i] + int(T/2)],
                          handY[i][dat['reach_times'][i] - int(T/2):dat['reach_times'][i] + int(T/2)]]).T

        hand = align_behavior(hand, T, bin_width)
        hand -= hand.mean(axis=0, keepdims=True)
        hand /= hand.std(axis=0, keepdims=True)
        
        dat['behavior'][i, ...] = hand

    return dat

def load_sabes_trialized(filename, min_length=6, **kwargs):

    # start time is handled in reach_segment_sabes, so do not prematurely truncate
    kwargs['truncate_start'] = False
    kwargs['segment'] = False
    # Load the data
    dat = load_sabes(filename, **kwargs)
    # Trialize
    dat_segment = reach_segment_sabes(dat, data_file=filename.split('/')[-1].split('.mat')[0])
    # Modfiy the spike rates and behavior entries according to the segmentation
    spike_rates = dat['spike_rates'].squeeze()
    spike_rates_trialized = [spike_rates[tt[0]:tt[1], :] 
                             for tt in dat_segment['transition_times']
                             if tt[1] - tt[0] > min_length]
    behavior = dat['behavior'].squeeze()
    behavior_trialized = [behavior[tt[0]:tt[1], :] for tt in dat_segment['transition_times']]
    dat['spike_rates'] = np.array(spike_rates_trialized, dtype=object)
    dat['behavior'] = np.array(behavior_trialized, dtype=object)
    return dat

def load_sabes(filename, bin_width=50, boxcox=0.5, filter_fn='none', filter_kwargs={}, spike_threshold=100,
               std_behavior=False, region='M1', high_pass=True, segment=False, return_wf=False, 
               subset=None, truncate_start=False, **kwargs):
    print('Starting loading')
    # Convert bin width to s
    bin_width /= 1000

    # Load MATLAB file
    # Avoid random OS errors
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']

        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
        dat = {}

        if region == 'M1':
            indices = M1_indices
        elif region == 'S1':
            indices = S1_indices
            print(len(indices))
        elif region == 'both':
            indices = list(range(n_channels))

        # Perform binning
        n_channels = len(indices)
        n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
        n_units = n_channels * n_sorted_units
        max_t = t[-1]

        spike_times = np.zeros((n_sorted_units - 1, len(indices))).astype(np.object)
        if return_wf:
            wf = np.zeros((n_sorted_units - 1, len(indices))).astype(np.object)

        for i, chan_idx in enumerate(indices):
            for unit_idx in range(1, n_sorted_units): # ignore hash
                spike_times_ = f[f["spikes"][unit_idx, chan_idx]][()]
                # Ignore this case (no data)
                if spike_times_.shape == (2,):
                    spike_times[unit_idx - 1, i] = np.nan
                else:
                    # offset spike times
                    spike_times[unit_idx - 1, i] = spike_times_[0, :] - t[0]
    
                if return_wf:
                    wf[unit_idx - 1, i] = f[f['wf'][unit_idx, chan_idx]][()].T

        # Reshape into format (ntrials, units)
        spike_times = spike_times.reshape((1, -1))
        if return_wf:
            wf = wf.reshape((1, -1))
        # Total length of the time series
        T = t[-1] - t[0]
        if return_wf:
            spike_rates, sufficient_spikes = postprocess_spikes(spike_times, T, bin_width, boxcox,
                                                                filter_fn, filter_kwargs, spike_threshold, high_pass=high_pass,
                                                                return_unit_filter=True)
            
            wf = wf[:, list(sufficient_spikes)]               
            dat['wf'] = wf
        else:
            spike_rates = postprocess_spikes(spike_times, T, bin_width, boxcox,
                                             filter_fn, filter_kwargs, spike_threshold, high_pass=high_pass)
        dat['spike_rates'] = spike_rates 

        # Get cursor position
        cursor_pos = f["cursor_pos"][:].T
        cursor_interp = align_behavior(cursor_pos, T, bin_width)
        if std_behavior:
            cursor_interp -= cursor_interp.mean(axis=0, keepdims=True)
            cursor_interp /= cursor_interp.std(axis=0, keepdims=True)

        dat["behavior"] = cursor_interp

        # Target position
        target_pos = f["target_pos"][:].T
        target_interp = align_behavior(target_pos, T, bin_width)
        # cursor_interp -= cursor_interp.mean(axis=0, keepdims=True)
        # cursor_interp /= cursor_interp.std(axis=0, keepdims=True)
        dat['target'] = target_interp

        dat['time'] = np.squeeze(align_behavior(t[:, np.newaxis], T, bin_width))

        # Pass through reach_segment_sabes and re-assign the behavior and spike_rates keys to the segmented versions 
        if segment:
            dat = reach_segment_sabes(dat, data_file=filename.split('/')[-1].split('.mat')[0])

            # Ensure we have somewhat long trajectories
            # T = 30
            # t = np.array([t_[1] - t_[0] for t_ in dat['transition_times']])
            # valid_transitions = np.arange(t.size)[t >= T]
            valid_transitions = np.arange(len(dat['transition_times']))
            spike_rates = np.array([dat['spike_rates'][0, dat['transition_times'][idx][0]:dat['transition_times'][idx][1]]
                                    for idx in valid_transitions])
            behavior = np.array([dat['behavior'][dat['transition_times'][idx][0]:dat['transition_times'][idx][1]]
                                 for idx in valid_transitions])

            dat['spike_rates'] = spike_rates
            dat['behavior'] = behavior
        
        if truncate_start:
            dat['spike_rates'] = dat['spike_rates'][:, start_times[filename.split('/')[-1].split('.mat')[0]]:]
            dat['behavior'] = dat['behavior'][start_times[filename.split('/')[-1].split('.mat')[0]]:]
        # Select a subset of neurons only
        if subset is not None:
            key = filename.split('/')[-1]
            if key not in subset:
                key = key.split('.mat')[0]
            dat['spike_rates'] = dat['spike_rates'][..., subset[key]]
        return dat

def load_sabes_wf(filename, spike_threshold=100, region='M1'):

    # Load MATLAB file
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']

        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
        dat = {}

        if region == 'M1':
            indices = M1_indices
        elif region == 'S1':
            indices = S1_indices
            print(len(indices))
        elif region == 'both':
            indices = list(range(n_channels))

        # Perform binning
        n_channels = len(indices)
        n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
        n_units = n_channels * n_sorted_units
        max_t = t[-1]

        spike_times = np.zeros((n_sorted_units - 1, len(indices))).astype(np.object)
        wf = np.zeros((n_sorted_units - 1, len(indices))).astype(np.object)

        for i, chan_idx in enumerate(indices):
            for unit_idx in range(1, n_sorted_units): # ignore hash
                spike_times_ = f[f["spikes"][unit_idx, chan_idx]][()]
                # Ignore this case (no data)
                if spike_times_.shape == (2,):
                    spike_times[unit_idx - 1, i] = np.nan
                else:
                    # offset spike times
                    spike_times[unit_idx - 1, i] = spike_times_[0, :] - t[0]

                wf[unit_idx - 1, i] = f[f['wf'][unit_idx, chan_idx]][()].T
            
        # # Reshape into format (ntrials, units)
        spike_times = spike_times.reshape((1, -1))
        wf = wf.reshape((1, -1))

        # # Apply spike threshold
        sizes = np.array([[1 if np.isscalar(spike_times[i, j]) else spike_times[i, j].size for j in range(spike_times.shape[1])]
                          for i in range(spike_times.shape[0])])

        sufficient_spikes = np.zeros(spike_times.shape).astype(np.bool_)
        for i in range(spike_times.shape[0]):
            for j in range(spike_times.shape[1]):
                if spike_threshold is not None:
                    if sizes[i, j] > spike_threshold:
                        sufficient_spikes[i, j] = 1
        
        wf = wf[sufficient_spikes]
        return wf


def load_peanut_across_epochs(fpath, epochs, spike_threshold, **loader_kwargs):

    dat_allepochs = {}
    dat_per_epoch = []

    unit_ids = []

    for epoch in epochs:
        dat = load_peanut(fpath, epoch, spike_threshold, **loader_kwargs)
        unit_ids.append(set(dat['unit_ids']))
        dat_per_epoch.append(dat)

    unit_id_intersection = unit_ids[0]
    for i in range(1, len(epochs)):
        unit_id_intersection.intersection(unit_ids[i])

    for i, epoch in enumerate(epochs):
        dat = dat_per_epoch[i]
        unit_idxs = np.isin(dat['unit_ids'], np.array(list(unit_id_intersection)).astype(int)) 

def load_peanut(fpath, epoch, spike_threshold, bin_width=25, boxcox=0.5,
                filter_fn='none', speed_threshold=4, region='HPc', filter_kwargs={}):
    '''
        Parameters:
            fpath: str
                 path to file
            epoch: list of ints
                which epochs (session) to load. The rat is sleeping during odd numbered epochs
            spike_threshold: int
                throw away neurons that spike less than the threshold during the epoch
            bin_width:  float 
                Bin width for binning spikes. Note the behavior is sampled at 25ms
            boxcox: float or None
                Apply boxcox transformation
            filter_fn: str
                Check filter_dict
            filter_kwargs
                keyword arguments for filter_fn
    '''

    data = pickle.load(open(fpath, 'rb'))
    dict_ = data['peanut_day14_epoch%d' % epoch]
    
    # Collect single units located in hippocampus

    HPc_probes = [key for key, value in dict_['identification']['nt_brain_region_dict'].items()
                  if value in ['HPc', 'HPC']]

    OFC_probes = [key for key, value in dict_['identification']['nt_brain_region_dict'].items()
                  if value == 'OFC']

    if region in ['HPc', 'HPC']:
        probes = HPc_probes
    elif region == 'OFC':
        probes = OFC_probes
    elif region == 'both':
        probes = list(set(HPc_probes).union(set(OFC_probes)))

    spike_times = []
    unit_ids = []
    for probe in dict_['spike_times'].keys():
        probe_id = probe.split('_')[-1]
        if probe_id in probes:
            for unit, times in dict_['spike_times'][probe].items():
                spike_times.append(list(times))
                unit_ids.append((probe_id, unit))
        else:
            continue


    # sort spike times
    spike_times = [list(np.sort(times)) for times in spike_times]

    # Apply spike threshold

    spike_threshold_filter = [idx for idx in range(len(spike_times))
                              if len(spike_times[idx]) > spike_threshold]
    spike_times = np.array(spike_times, dtype=object)
    spike_times = spike_times[spike_threshold_filter]
    unit_ids = np.array(unit_ids)[spike_threshold_filter]

    t = dict_['position_df']['time'].values
    T = t[-1] - t[0] 
    # Convert bin width to s
    bin_width = bin_width/1000
    
    # covnert smoothin bandwidth to indices
    if filter_fn == 'gaussian':
        filter_kwargs['sigma'] /= bin_width
        filter_kwargs['sigma'] = min(1, filter_kwargs['sigma'])
    
    bins = np.linspace(0, T, int(T//bin_width))

    spike_rates = np.zeros((bins.size - 1, len(spike_times)))
    for i in range(len(spike_times)):
        # translate to 0
        spike_times[i] -= t[0]
        
        spike_counts = np.histogram(spike_times[i], bins=bins)[0]
        if boxcox is not None:
            spike_counts = np.array([(np.power(spike_count, boxcox) - 1)/boxcox
                                     for spike_count in spike_counts])
        spike_rates_ = FILTER_DICT[filter_fn](spike_counts.astype(np.float), **filter_kwargs)
        
        spike_rates[:, i] = spike_rates_
    
    # Align behavior with the binned spike rates
    pos_linear = dict_['position_df']['position_linear'].values
    pos_xy = np.array([dict_['position_df']['x-loess'], dict_['position_df']['y-loess']]).T
    pos_linear, taligned = align_peanut_behavior(t, pos_linear, bins)
    pos_xy, _ = align_peanut_behavior(t, pos_xy, bins)
    
    dat = {}
    dat['unit_ids'] = unit_ids
    # Apply movement threshold
    if speed_threshold is not None:
        vel = np.divide(np.diff(pos_linear), np.diff(taligned))
        # trim off first index to match lengths
        spike_rates = spike_rates[1:, ...]
        pos_linear = pos_linear[1:, ...]
        pos_xy = pos_xy[1:, ...]

        spike_rates = spike_rates[np.abs(vel) > speed_threshold]

        pos_linear = pos_linear[np.abs(vel) > speed_threshold]
        pos_xy = pos_xy[np.abs(vel) > speed_threshold]

    dat['unit_ids'] = unit_ids
    dat['spike_rates'] = spike_rates
    dat['behavior'] = pos_xy
    dat['behavior_linear'] = pos_linear[:, np.newaxis]
    dat['time'] = taligned
    return dat

##### Peanut Segmentation #####
def segment_peanut(dat, loc_file, epoch, box_size=20, start_index=0, return_maze_points=False):

    with open(loc_file, 'rb') as f:
        ldict = pickle.load(f)
        
    edgenames = ldict['peanut_day14_epoch2']['track_graph']['edges_ordered_list']
    nodes = ldict['peanut_day14_epoch%d' % epoch]['track_graph']['nodes']
    for key, value in nodes.items():
        nodes[key] = (value['x'], value['y'])
    endpoints = []
    lengths = []
    for edgename in edgenames:
        endpoints.append(ldict['peanut_day14_epoch%d' % epoch]['track_graph']['edges'][edgename]['endpoints'])
        lengths.append(ldict['peanut_day14_epoch%d' % epoch]['track_graph']['edges'][edgename]['length'])
        
    # pos = np.array([ldict['peanut_day14_epoch%d' % epoch]['position_input']['position_x'],
    #             ldict['peanut_day14_epoch%d' % epoch]['position_input']['position_y']]).T
    pos = dat['behavior']
    if epoch in [2, 6, 10, 14]:
        transition1 = find_transitions(pos, nodes, 'handle_well', 'left_well', 
                                                   ignore=['center_maze', 'left_corner'], box_size=box_size, start_index=start_index)
        transition2 = find_transitions(pos, nodes, 'handle_well', 'right_well',
                                                   ignore=['center_maze', 'right_corner'], box_size=box_size, start_index=start_index)
    elif epoch in [4, 8, 12, 16]:
        transition1 = find_transitions(pos, nodes, 'center_well', 'left_well', 
                                                   ignore=['center_maze', 'left_corner'], box_size=box_size, start_index=start_index)
        transition2 = find_transitions(pos, nodes, 'center_well', 'right_well',
                                                   ignore=['center_maze', 'right_corner'], box_size=box_size, start_index=start_index)
    if return_maze_points:
        return transition1, transition2, nodes, endpoints
    else:
        return transition1, transition2

def in_box(pos, node, box_size):
    box_points = [np.array(node) + box_size/2 * np.array([1, 1]), # Top right
                  np.array(node) + box_size/2 * np.array([1, -1]), # Bottom right
                  np.array(node) + box_size/2 * np.array([-1, 1]), # Top left
                  np.array(node) + box_size/2 * np.array([-1, -1])] # Bottom left

    in_xlim = np.bitwise_and(pos[:, 0] > box_points[-1][0], 
                             pos[:, 0] < box_points[0][0])
    in_ylim = np.bitwise_and(pos[:, 1] > box_points[-1][1], 
                             pos[:, 1] < box_points[0][1])    
    return np.bitwise_and(in_xlim, in_ylim)
    
def find_transitions(pos, nodes, start_node, end_node, ignore=['center_maze'],
                     box_size=20, start_index=1000):
    pos = pos[start_index:]
    
    in_node_boxes = {}
    for key, value in nodes.items():
        in_node_boxes[key] = in_box(pos, value, box_size)
        
    in_node_boxes_windows = {}
    for k in in_node_boxes.keys():
        in_node_boxes_windows[k] = [[i for i,value in it] 
                                    for key,it in 
                                    itertools.groupby(enumerate(in_node_boxes[k]), key=operator.itemgetter(True)) 
                                    if key != 0]

    # For each window of time that the rat is in the start node box, find which box it goes to next. If this
    # box matches the end_node, then add the intervening indices to the list of transitions
    transitions = []
    for start_windows in in_node_boxes_windows[start_node]:
        next_box_times = {}
        
        # When does the rat leave the start_node
        t0 = start_windows[-1]
        for key, windows in in_node_boxes_windows.items():
            window_times = np.array([time for window in windows for time in window])
            # what is the first time after t0 that the rat enters this node/box
            valid_window_times = window_times[window_times > t0]
            if len(valid_window_times) > 0:
                next_box_times[key] = window_times[window_times > t0][0]
            else:
                next_box_times[key] = np.inf

        # Order the well names by next_box_times
        node_names = list(next_box_times.keys())
        node_times = list(next_box_times.values())
        
        
        node_order = np.argsort(node_times)
        idx = 0
        # Find the first node that is not the start_node and is not in the list of nodes to ignore
        while (node_names[node_order[idx]] in ignore) or (node_names[node_order[idx]] == start_node):
            idx += 1

        if node_names[node_order[idx]] == end_node:
            # Make sure to translate by the start index
            transitions.append(np.arange(t0, node_times[node_order[idx]]) + start_index)
            
    return transitions

# Segment the time series and then use the linearized positions to calculate the occupancy normalized firing rates, binned by position
def location_bin_peanut(fpath, loc_file, epoch, spike_threshold=100, sigma = 2):

    # No temporal binning
    dat = load_peanut(fpath, epoch, spike_threshold=spike_threshold, bin_width=1, boxcox=None,
                      speed_threshold=0)

    transition1, transition2 = segment_peanut(dat, loc_file, epoch)
    occupation_normed_rates = []
    transition_bins = []
    for transition_ in [transition1, transition2]:

        # Concatenate position and indices
        pos = []
        indices = []
        for trans in transition_:
            pos.extend(list(dat['behavior_linear'][trans, 0]))
            indices.extend(trans)
        indices = np.array(indices)

        bins = np.linspace(min(pos), max(pos), int((max(pos) - min(pos))/2))
        transition_bins.append(bins)
        # Histogram the linearized positions into the bins
        occupation_counts, _, idxs = binned_statistic(pos, pos, statistic='count', bins=bins)

        # Sum up spike counts
        binned_spike_counts = np.zeros((len(occupation_counts), dat['spike_rates'].shape[1]))
        for j in range(len(occupation_counts)):
            bin_idxs = indices[np.where(idxs==j + 1)[0]]
            for k in range(dat['spike_rates'].shape[1]):
                binned_spike_counts[j, k] = np.sum(dat['spike_rates'][bin_idxs, k])  

        # Smooth occupation_cnts and binned_spike_counts by a Gaussian filter of width 2 indices (4 cm)
        smooth_occupation_counts = gaussian_filter1d(occupation_counts, sigma=sigma)
        smooth_binned_rates = gaussian_filter1d(binned_spike_counts, sigma=sigma, axis=0)
        # Normalize binned_spike_counts by occupation_counts
        smooth_binned_rates = np.divide(smooth_binned_rates, smooth_occupation_counts[:, np.newaxis])

        # Set units to hertz
        dt = dat['time'][1] - dat['time'][0]
        smooth_binned_rates /= dt
        occupation_normed_rates.append(smooth_binned_rates)

    return occupation_normed_rates, transition_bins

def load_shenoy_large(path, bin_width=50, boxcox=0.5, trialize=False, filter_fn='none', filter_kwargs={}, spike_threshold=100,
                      trial_threshold=0.5, std_behavior=False, location='M1', interval='full'):

    local_args = locals()
    # Convert bin width to s
    bin_width /= 1000
    path_root = path.split('.nwb')[0]
    tmp_files = glob.glob(path_root + '_tmp*')
    using_tmp = False
    tmp_indices = []
    for tmp_file in tmp_files:
        tmp_index = int(tmp_file.split('tmp')[1].split('.pkl')[0])
        tmp_indices.append(tmp_index)

        with open(tmp_file, 'rb') as f:
            try:
                loader_args = pickle.load(f)         
            except:
                continue
        all_close = True
        for k, v in loader_args.items():
            if k == 'path':
                continue
            if local_args[k] != v:
                all_close = False
                break

        if all_close:
            using_tmp = True
            print('Using already loaded version of the file!')
            break

    if not using_tmp:
        # Check to see whether the file has already been loaded with the desired loader args
        io = NWBHDF5IO(path, 'r')
        nwbfile_in = io.read()
        # Get successful trial indices
        valid_trial_indices = set(list(nwbfile_in.trials.task_success[:].nonzero()[0]))
        discard_invert = set(list(np.invert(nwbfile_in.trials.discard_trial[:]).nonzero()[0]))
        valid_trial_indices = valid_trial_indices.intersection(discard_invert)
        valid_trials = np.array(list(valid_trial_indices))

        # Need to restrict to trials where there is a non-zero delay period prior to go cue
        if interval == 'before_go':
            valid_trials_ = []
            for trial in valid_trials:
                if nwbfile_in.trials.go_cue_time[trial] - nwbfile_in.trials.start_time[trial] > 2 * bin_width:
                    valid_trials_.append(trial)
            valid_trials = np.array(valid_trials_)

        print('%d valid trials' % len(valid_trials))

        # Get index of electrodes located in the desired area
        loc_dict = {'M1':'M1 Motor Cortex', 'PMC': 'Pre-Motor Cortex, dorsal'}
        valid_units = []
        for i, loc in enumerate(nwbfile_in.electrodes.location):
            if loc == loc_dict[location]:
                valid_units.append(i)

        print('%d valid  units' % len(valid_units))
        raw_spike_times = np.array(nwbfile_in.units.spike_times_index)
        if trialize:
            # Trialize spike_times
            spike_times = np.zeros((len(valid_trials), len(valid_units)), dtype=np.object)
            T = np.zeros((valid_trials.size, 2))
            print('Trializing spike times')
            for j, unit in tqdm(enumerate(valid_units)):
                end_idx = 0
                for i, trial in enumerate(valid_trials):
                    if interval == 'full':
                        T[i, 0] = nwbfile_in.trials.start_time[trial]
                        T[i, 1] = nwbfile_in.trials.stop_time[trial]
                    elif interval == 'before_go':
                        T[i, 0] = nwbfile_in.trials.start_time[trial]
                        T[i, 1] = nwbfile_in.trials.go_cue_time[trial]
                    elif interval == 'after_go':
                        T[i, 0] = nwbfile_in.trials.go_cue_time[trial]
                        T[i, 1] = nwbfile_in.trials.stop_time[trial]
                    else:
                        raise ValueError('Invalid interval, please specify full, before_go, or after_go')
                    windowed_spike_times, end_idx = window_spikes(raw_spike_times[unit], 
                                                                T[i][0], T[i][1], end_idx)
                    spike_times[i, j] = windowed_spike_times

            T = np.squeeze(np.diff(T, axis=1))
        else:
            spike_times = np.zeros((1, len(valid_units)), dtype=np.object)
            for j, unit in enumerate(valid_units):
                spike_times[0, j] = raw_spike_times[unit]
            T = nwbfile_in.units.obs_intervals[0][1]

        # Filter spikes
        spike_rates = postprocess_spikes(spike_times, T, bin_width, boxcox, filter_fn, filter_kwargs,
                                         spike_threshold, trial_threshold)
        dat = {}
        if trialize:
            dat['spike_rates'] = np.array(spike_rates, dtype=object)
        else:
            dat['spike_rates'] = np.array(spike_rates).squeeze()            
        if trialize:
            dat['target_pos'] = np.array([nwbfile_in.trials.target_positions_index[trial] for trial in valid_trials])
        else:
            dat['target_pos'] = np.array([nwbfile_in.trials.target_positions_index])
        
        # Return go_cue_times relative to start_time
        dat['go_times'] = nwbfile_in.trials.go_cue_time[valid_trials] - nwbfile_in.trials.start_time[valid_trials]

        t = nwbfile_in.processing['behavior'].data_interfaces['Position'].spatial_series['Cursor'].timestamps

        if trialize:
            # Trialize behavior
            cursor = np.zeros(len(valid_trials), dtype=np.object) 
            hand = np.zeros(len(valid_trials), dtype=np.object) 

            t = nwbfile_in.processing['behavior'].data_interfaces['Position'].spatial_series['Cursor'].timestamps

            print('Trializing Behavior')
            # Seems funky here - also need to inspect what non-trialized means
            for i, trial in tqdm(enumerate(valid_trials)):
                if interval == 'full':
                    start_index = np.where(t > nwbfile_in.trials.start_time[trial])[0][0]
                    end_index = np.where(t < nwbfile_in.trials.stop_time[trial])[0][-1]
                elif interval == 'before_go':
                    start_index = np.where(t > nwbfile_in.trials.start_time[trial])[0][0]
                    end_index = np.where(t < nwbfile_in.trials.go_cue_time[trial])[0][-1]
                elif interval == 'after_go':
                    start_index = np.where(t > nwbfile_in.trials.go_cue_time[trial])[0][0]
                    end_index = np.where(t < nwbfile_in.trials.stop_time[trial])[0][-1]
                else:
                    raise ValueError('Invalid interval, please specify full, before_go, or after_go')
                cursor_ = nwbfile_in.processing['behavior'].data_interfaces['Position'].spatial_series['Cursor'].data[start_index:end_index]
                cursor[i] = align_behavior(cursor_, T[i], bin_width)
                hand_ = nwbfile_in.processing['behavior'].data_interfaces['Position'].spatial_series['Hand'].data[start_index:end_index]
                hand[i] = align_behavior(hand_, T[i], bin_width)
            # Align behavior    
            #cursor_interp = np.array([align_behavior(c, T[i], bin_width) for i, c in enumerate(cursor)], dtype=np.object)
            #hand_interp = np.array([align_behavior(h, T[i], bin_width) for i, h in enumerate(hand)], dtype=np.object)
            cursor_interp = cursor
            hand_interp = hand
        else:
            cursor = nwbfile_in.processing['behavior'].data_interfaces['Position'].spatial_series['Cursor'].data
            hand = nwbfile_in.processing['behavior'].data_interfaces['Position'].spatial_series['Hand'].data

            cursor_interp = align_behavior(cursor, T, bin_width)
            hand_interp = align_behavior(cursor, T, bin_width)

        dat['behavior'] = np.array(cursor_interp)
        dat['behavior_3D'] = np.array(hand_interp)

        io.close()

        # Save the processed data to disk so we can save time with similar loads in the future
        if len(tmp_indices) > 0:
            tmp_index = max(tmp_indices) + 1
        else:
            tmp_index = 0
        path = path.split('.nwb')[0]
        with open(path + '_tmp%d.pkl' % tmp_index, 'wb') as f:
            f.write(pickle.dumps(local_args))
            f.write(pickle.dumps(dat))
    else:
        with open(tmp_file, 'rb') as f:
            _ = pickle.load(f)
            dat = pickle.load(f)
    return dat

def load_tsao(data_path, bin_width=25, boxcox=0.5, filter_fn='none', filter_kwargs={},  region="both", spike_threshold=None, trial_threshold=0, degraded=True, verbose=True, manual_unit_selection=[], same_trial_dur=False):
     
    #""" Regions tags: "ML", "AM" """

    """Check to see if these arguments have already been loaded on this dataset. If so, load and return that loaders output (dat struct below)"""
    arg_dict = {
        'data_path': data_path, 'bin_width': bin_width, 'boxcox': boxcox, 'degraded':degraded,
        'filter_fn': filter_fn, 'filter_kwargs': make_hashable(filter_kwargs),
        'region': region, 'spike_threshold': spike_threshold, 'trial_threshold': trial_threshold,
        'manual_unit_selection': tuple(manual_unit_selection), 'same_trial_dur':same_trial_dur
        }
    arg_tuple = tuple(sorted(arg_dict.items()))
    
    

    # Overwrite data path 
    if data_path[-6:] == 'pickle':
        preload_dict_path = '/'.join(data_path.split('/')[:-1]) + '/preloaded/preloadDict.pickle'
    else:
        preload_dict_path = data_path + '/preloaded/preloadDict.pickle'

    with open(preload_dict_path, 'rb') as file:
        preloadDict = pickle.load(file)

    # If these args have been run on this data set, load that data
    for args in preloadDict.keys():
        if np.all([x == y for (x, y) in zip(args, arg_tuple)]):
            print("Preloading data...")
            preloadID = preloadDict[arg_tuple]
            loaded_data_path = os.path.dirname(preload_dict_path) + f"/preloaded_data_{preloadID}.pickle"
            with open(loaded_data_path, 'rb') as file:
                dat = pickle.load(file)
            return dat


    """"Otherwise, load the data "fresh", keeping in mind that loaders assumes that the Tsao structures have been split by paradigm into a .pickle file, which is here loaded"""
    if verbose: print("Begin Loading Data...")
    
    with open(data_path, 'rb') as file:
        f = pickle.load(file)
    if verbose: print("Done Loading Data")


    if region == "both":
        regionINDs = np.arange(len(f["regionIDs"]))
        regions = f["regions"]
    else:
        regionID = f["regions"].index(region)
        regionINDs = np.where(regionID == f["regionIDs"])[0]

        regions = f["regions"][regionID]
        regionIDs = f["regionIDs"][regionINDs]

    # This selects units for all downstream analyses, since regionINDs selects units from the raw input data. 
    if len(manual_unit_selection) != 0:
        regionIDs = regionIDs[list(manual_unit_selection)]
        regionINDs = regionINDs[list(manual_unit_selection)]

    # Get useful details from the file, used for computing spike rates
    TrialStartTimes = f['timecourse_on_times'] # Units of ms, relative to paradigm start stime
    StimONDurations = f['on_durations'] 
    StimOFFDurations = f['off_durations'] 
    StimulusNames = f['condition_stimulus_list'] 
    StimulusIndicies = f['stimulus_index_valid'] 
    BinarySpikeTimes = f['timecourse_all_units'][:, regionINDs]
    
    # This selects trials for all downstream analyses
    if len(manual_trial_selection) != 0:
        TrialStartTimes = TrialStartTimes[list(manual_trial_selection)]
        StimONDurations = StimONDurations[list(manual_trial_selection)]
        StimOFFDurations = StimOFFDurations[list(manual_trial_selection)]
        StimulusIndicies = StimulusIndicies[list(manual_trial_selection)]

    
    numUnits = np.shape(BinarySpikeTimes)[1]
    numTrials = len(TrialStartTimes)
    SpikeMats = np.empty(shape=(numTrials,numUnits), dtype='object')
    TrialDurations = StimONDurations + StimOFFDurations

    # Get spike times 
    if verbose: print("Begin getting spike times...")
    for trialInd, trialStartTime in enumerate(TrialStartTimes):

        # Get the relevant section of the response matrix
        spikes = BinarySpikeTimes[int(trialStartTime):int(trialStartTime+TrialDurations[trialInd]), :]

        for unit in range(numUnits):
            # for each neuron, convert binary spike vector to spike times relative to trial onset
            spikeTimes = np.nonzero(spikes[:,unit]) # units of ms
            SpikeMats[trialInd, unit] = spikeTimes
    if verbose: print("Done getting spike times")


    # Filter spikes
    if verbose: print("Begin filtering spike times into spike rates...")
    
    if same_trial_dur:
        T = max(TrialDurations)
    else:
        T = TrialDurations

    spike_rates = postprocess_spikes(SpikeMats, T, bin_width, boxcox, filter_fn, dict(filter_kwargs), spike_threshold=spike_threshold, trial_threshold=trial_threshold)
    if verbose: print("Done filtering spike times into spike rates")


    dat = {}
    dat["ParadigmName"] = f["ParadigmName"] 
    dat["probeID"] = f["probeID"] 
    dat["channelID"] = f["channelID"][regionINDs] 
    dat["waveforms"] = f["waveforms"][regionINDs] 
    dat["waveforms_time"] = f["waveforms_time"]
    dat["Regions"] = regions
    dat["regionIDs"] = regionIDs
    dat["NumUnits"] = numUnits
    dat["NumTrials"] = numTrials
    dat["TrialStartTimes"] = TrialStartTimes
    dat["StimulusOnDurations"] = StimONDurations
    dat["StimulusOFFDurations"] = StimOFFDurations
    dat["StimulusNames"] = StimulusNames
    dat["StimIDs"] = StimulusIndicies
    dat["behavior"] = StimulusIndicies
    dat["TrialDurations"] = TrialDurations
    dat["bin_width"] = bin_width
    dat["SessionName"] = os.path.splitext(os.path.basename(data_path))[0]
    dat["spike_times"] = SpikeMats
    dat["spike_rates"] = spike_rates
    dat["full_arg_tuple"] = arg_tuple


    if degraded:
        degrade_types = ["blur", "contrast", "mooney_gray", "noise", "banana", "body", "bottle", "box", "rand1", "rand2"]
        stimulus_ID_from_names = np.arange(1, len(StimulusNames)+1)

        degraded_stim_inds = np.zeros(len(StimulusNames))
        for idx, name in enumerate(StimulusNames):
            for deg in degrade_types:
                if deg in name:
                    degraded_stim_inds[idx] = 1

        degradedIDs = stimulus_ID_from_names[degraded_stim_inds.astype(bool)]
        #degraded_trials = np.zeros(len(StimulusIndicies))
        degraded_trials = np.array([ ID in degradedIDs for ID in StimulusIndicies]).astype(int)

        dat["stratifiedIDs"] = degraded_trials
        dat["degradedIDs"] = degradedIDs

    else:
        dat["stratifiedIDs"] = StimulusIndicies


    """This section saves the loaded data to the preloaded folder, and gives this loader call a unique ID that indexes all the loader calls for this dataset"""
    # Assign an ID to this loader call
    if not preloadDict:
        preloadID = 0
    else:
        preloadID = max(list(preloadDict.values())) + 1
    # Add this ID and the arugemnts for the loader to the preload dict
    preloadDict[arg_tuple] = preloadID


    # Save the preload dict and the actual data
    with open(preload_dict_path, 'wb') as file:
        pickle.dump(preloadDict, file)

    loaded_data_path = os.path.dirname(preload_dict_path) + f"/preloaded_data_{preloadID}.pickle"
    with open(loaded_data_path, 'wb') as file:
        pickle.dump(dat, file)


    return dat


def load_AllenVC(data_path, region="VISp", bin_width=25, preTrialWindowMS=50, postTrialWindowMS=100, boxcox=0.5, filter_fn='none', filter_kwargs={}, spike_threshold=None, trial_threshold=0):

    # Loads one session at a time
    
    # ------------------------------- Check if these params have already been applied/loaded first, or load new ::
    
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)    
    arg_dict = {arg: values[arg] for arg in args}
    arg_dict['filter_kwargs'] = make_hashable(arg_dict['filter_kwargs'])    
    arg_tuple = tuple(sorted(arg_dict.items()))
    
    DataFolderPath = os.path.dirname(os.path.dirname(data_path))
    preload_dict_path = DataFolderPath + '/preloaded/preloadDict.pickle'
    with open(preload_dict_path, 'rb') as file:
        preloadDict = pickle.load(file)

    for args in preloadDict.keys():
        if args == arg_tuple:
            print("Preloading data...")
            preloadID = preloadDict[arg_tuple]
            loaded_data_path = os.path.dirname(preload_dict_path) + f"/preloaded_data_{preloadID}.pickle"
            with open(loaded_data_path, 'rb') as file:
                dat = pickle.load(file)
            return dat


    # ------------------------------- Otherwise, load the data "fresh"    
    print("Begin Loading Data Fresh ...")
    

    # Get Allen structures for loading data
    manifest_path = os.path.join(DataFolderPath, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    session_id = int(os.path.splitext(os.path.basename(data_path))[0].split('_')[1])

    
    # For all session info, including regions, stimulus names, unit count, etc. see: session.metadata
    warnings.filterwarnings("ignore", category=UserWarning)
    session = cache.get_session_data(session_id)
        
    units = session.units[session.units["ecephys_structure_acronym"] == region]
    if units.empty: return {} # Check that this region is in this session and has units

    presentations = session.get_stimulus_table("natural_scenes") 
    stimIDs = presentations.loc[:, "frame"].values.astype(int) # Per trial stimulus IDs

    


    # Pre-, and post- trial windows are in units of ms. Convert to seconds
    binarize_bin = 1/1000 # 1ms bins in units of seconds
    DefaultTrialDuration = 0.25 # units of seconds
    time_bins = np.arange(-(preTrialWindowMS/1000), DefaultTrialDuration + (postTrialWindowMS/1000) + binarize_bin, binarize_bin)

    histograms = session.presentationwise_spike_counts(
        stimulus_presentation_ids=presentations.index.values,  
        bin_edges=time_bins,
        unit_ids=units.index.values)
    
    binary_spikes = np.array(histograms) # trial, time, unit. use 'histograms.coords' to confirm
    
    
    # Given a binary spike matrix, get spike times.
    numTrials, numTimePoints, numUnits = binary_spikes.shape

    SpikeMats = np.empty((numTrials, numUnits), dtype='object')
    for trial in range(numTrials):
        for unit in range(numUnits):
            SpikeMats[trial, unit] = np.where(binary_spikes[trial, :, unit] != 0)[0]
    
    # SpikeMats reports for each (trial, unit) the time in ms of a spike
    # RELATIVE TO "preTrialWindow" seconds before the trial starts, and until "postTrialWindow" seconds after the trial starts
    
    
    T = numTimePoints # units of ms duration of a trial (here, includes pre- and post- windows)
    spike_rates = postprocess_spikes(SpikeMats, T, bin_width, boxcox, filter_fn, dict(filter_kwargs), spike_threshold=spike_threshold, trial_threshold=trial_threshold)
    
    
    
    dat = {}
    dat["spike_rates"] = spike_rates
    dat["behavior"] = stimIDs
    dat["preTrialWindow"] = preTrialWindowMS
    dat["postTrialWindow"] = postTrialWindowMS
    dat["spike_times"] = SpikeMats



    # ------------------------------- Save this data run for the future

    # Assign an ID to this loader call
    if not preloadDict: preloadID = 0
    else: preloadID = max(list(preloadDict.values())) + 1
    preloadDict[arg_tuple] = preloadID 

    # Save the preload dict and the actual data
    with open(preload_dict_path, 'wb') as file:
        pickle.dump(preloadDict, file)

    loaded_data_path = os.path.dirname(preload_dict_path) + f"/preloaded_data_{preloadID}.pickle"
    with open(loaded_data_path, 'wb') as file:
        pickle.dump(dat, file)

    
    return dat


def make_hashable(d):
    """ Recursively convert a dictionary into a hashable type (tuples of tuples). """
    if isinstance(d, dict):
        return tuple((key, make_hashable(value)) for key, value in sorted(d.items()))
    elif isinstance(d, list):
        return tuple(make_hashable(value) for value in d)
    else:
        return d

def load_franklab_new(path, session=None, bin_width=50, boxcox=0.5, filter_fn='none', filter_kwargs={}, spike_threshold=100,
                      region='mPFC', speed_threshold=True, trialize=False):
    if speed_threshold and trialize:
        raise NotImplementedError
    
    #if session is not None:
    #    print("Need to implement multi-session recordinds for Frank Lab later !")

    # convert to seconds
    bin_width /= 1000

    with open('%s/spikes_filt.pkl' % path, 'rb') as f:
        spikes_HPC = pickle.load(f)
        spikes_OFC = pickle.load(f)
        spikes_mPFC = pickle.load(f)

    with open('%s/position.pkl' % path, 'rb') as f:
        position_df = pickle.load(f)

    if region == 'HPC':
        spikes = spikes_HPC
    elif region == 'OFC':
        spikes = spikes_OFC
    elif region == 'mPFC':
        spikes = spikes_mPFC

    # remove nan times from position df 
    mask = position_df['position_x'].notna()
    # apply mask
    position_df = position_df[mask]

    # Make sure all spikes occur in the period of time for 
    # which we have position data
    min_t = min(position_df.index.values)
    max_t = max(position_df.index.values)
    valid_spikes = []
    for i, n in enumerate(spikes):
        valid_spikes.append([])
        for spike in n:
            if spike > min_t and spike < max_t:
                # Include the spike and shift time to measure by 0
                valid_spikes[i].append(spike - min_t)
    valid_spikes = np.array(valid_spikes, dtype=object)
    # add a trial dimension for compatibility with postprocess_spikes
    valid_spikes = valid_spikes[np.newaxis, :]
    dat = {}
    # get firing rates

    # Total length of time series
    T = max_t - min_t
    spike_rates = postprocess_spikes(valid_spikes, T, bin_width, boxcox,
                                     filter_fn, filter_kwargs, spike_threshold)

    # remove trial dimension
    spike_rates = spike_rates.squeeze()
    # Get cursor position
    pos = np.hstack([position_df['position_x'].values[:, np.newaxis],
                     position_df['position_y'].values[:, np.newaxis]])
    vel = np.hstack([position_df['velocity_x'].values[:, np.newaxis],
                     position_df['velocity_y'].values[:, np.newaxis]])
    speed = position_df['speed'].values[:, np.newaxis]
    orientation = position_df['orientation'].values[:, np.newaxis]

    pos_interp = align_behavior(pos, T, bin_width)
    vel_interp = align_behavior(vel, T, bin_width)
    speed_interp = align_behavior(speed, T, bin_width)
    orient_interp = align_behavior(orientation, T, bin_width)
    # currently do not support both a speed threshold and trialization
    if speed_threshold:
        valid_indices = np.where(speed_interp.squeeze() > speed_threshold)[0]
        speed_interp = speed_interp[valid_indices]
        pos_interp = pos_interp[valid_indices]
        vel_interp = vel_interp[valid_indices]
        orient_interp = orient_interp[valid_indices] 
        spike_rates = spike_rates[valid_indices]

    dat["behavior"] = pos_interp
    dat['velocity'] = vel_interp
    dat['speed'] = speed_interp
    dat['orientation'] = orient_interp
    dat['spike_rates'] = spike_rates 

    if trialize:
        start_indices, end_indices = trialize_franklab(R=15) # Box size to use around wells. R=15 is good. See FrankLab_trialization.ipynb to visualize.
        # Convert to raw times
        start_times = [position_df.index[s] for s in start_indices]
        end_times = [position_df.index[e] for e in end_indices]

        # convert start and end times back into indices compatible with the binned time series
        binned_time = align_behavior(position_df.index.values[:, np.newaxis], T, bin_width)
        start_indices = [np.argmin(np.abs(binned_time.squeeze() - s)) for s in start_times]
        end_indices = [np.argmin(np.abs(binned_time.squeeze() - e)) for e in end_times]
        # Each entry corresponds to a trial, and each trial is of shape (time, unit) where each "trial" will have a different time.
        # At least 1 s trial
        min_length = int(1/bin_width)
        spike_rates = np.array([dat['spike_rates'][start:end, :] for start,end in zip(start_indices, end_indices) 
                                if end - start > min_length], dtype=object)
        lengths = [s.shape[0] for s in spike_rates]
        behavior = np.array([dat['behavior'][start:end, :] for start,end in zip(start_indices, end_indices) 
                             if end - start > min_length], dtype=object)
        # Overwrite dat
        dat['spike_rates'] = np.array(spike_rates) 
        dat['behavior'] = np.array(behavior) 

    return dat



def load_organoids(data_path, trial_length=2000, bin_width=50, boxcox=0.5, filter_fn='none', filter_kwargs={}, spike_threshold=0):

    """
    'trial_length' will trialize the data into windows of this length (units of ms)
    """
    print_stuff = False
    if print_stuff: print('Loading Data...')

    # Load AllSessions
    data_file = "/home/marcush/Data/OrganoidData/AllSessions.pkl"
    with open(data_file, "rb") as f:
        AllSessions = pickle.load(f)

    
    unitIDs = list(AllSessions.keys())
    numUnits = len(unitIDs)
    all_spike_times = [time for times in AllSessions.values() for time in times]
    n_trials = (max(all_spike_times) + 1) // trial_length  


    if print_stuff: print('Trializing Data...')

    # Initialize SpikeMats (n_trials x n_units) where each element is a list
    SpikeMats = np.empty((n_trials, numUnits), dtype='object')
    for trial in range(n_trials):
        for unit in range(numUnits):
            unit_spikes = np.array(AllSessions[unitIDs[unit]])  # Spike times for this unit
            trial_start = trial * trial_length
            trial_end = trial_start + trial_length
            
            # Keep only spikes that fall within this trial window, and subtract trial start
            relative_spikes = unit_spikes[(unit_spikes >= trial_start) & (unit_spikes < trial_end)] - trial_start
            
            SpikeMats[trial, unit] = relative_spikes

    if print_stuff: print('Postprocessing Spikes...')
    T = trial_length 
    spike_rates = postprocess_spikes(SpikeMats, T, bin_width, boxcox, filter_fn, filter_kwargs, spike_threshold)
  
  
    dat = {}
    dat['spike_rates'] = np.array(spike_rates) 
    dat['unitIDs'] = unitIDs
    dat['numUnits'] = numUnits
    dat['behavior'] = np.array([])

    if print_stuff: print('Done Loading Data.')

    return dat



def is_in_well(x, y, center, R):
    cx, cy = center
    return (cx - R <= x <= cx + R) and (cy - R <= y <= cy + R)

def trialize_franklab(R=15):

    # Load the position dataframe
    path = '/clusterfs/NSDS_data/franklabdata/dataset1'
    with open('%s/position.pkl' % path, 'rb') as f:
            position_df = pickle.load(f)


    mask = position_df['position_x'].notna()
    position_df = position_df[mask]
    x_positions = np.array(position_df['position_x'])
    y_positions = np.array(position_df['position_y'])

    # Reward/objective positions retrieved from Frank lab directly
    centers = {
    'maze_end_1': (79.910, 216.720),  # Top-left arm
    'maze_end_2': (92.693, 42.345),   # Bottom-left arm
    'maze_end_3': (31.340, 126.110),  # Middle-left arm
    'maze_end_4': (183.718, 217.713), # Top-right arm
    'maze_end_5': (183.784, 45.375),  # Bottom-right arm
    'maze_end_6': (231.338, 136.281)  # Middle-right arm
    }

    # Given a box of radius R around the wells, find times when the mouse enters/exits the box
    all_in_well = []
    for x,y in zip(x_positions, y_positions):
        in_any_well = False 

        for well, center in centers.items():
            if is_in_well(x, y, center, R):
                in_any_well = True
        all_in_well.append(in_any_well)

    all_in_well = np.array(all_in_well).astype(int)

    changes = np.diff(all_in_well) # a 1 means they've entered a well, a -1 means they've left a well
    start_times = np.squeeze(np.argwhere(changes == -1)) # Therefore, a trial begins when they leave a well
    end_times = np.squeeze(np.argwhere(changes == 1))    # And that trial ends when they enter the next well

    # Clean it up, make sure it fits.
    if start_times[0] > end_times[0]:
        end_times = end_times[1:]
    if start_times.size > end_times.size:
        start_times = start_times[:end_times.size]
    elif end_times.size > start_times.size:
        end_times = end_times[:start_times.size]


    return start_times, end_times