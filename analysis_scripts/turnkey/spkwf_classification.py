import numpy as np
import glob
import scipy
import os
import pandas as pd
import pdb
from tqdm import tqdm
import pickle
import sys
from matplotlib import pyplot as plt
import h5py

import sys
from config import *
from region_select import *
sys.path.append(PATH_DICT['repo'])

from loaders import load_sabes_wf, load_sabes
from utils import apply_df_filters, calc_loadings
from segmentation import reach_segment_sabes    

# Extract the spike width and hyperpolarization
def extract_wf_features(spikes):

    spikes_pp = []
    for spike in spikes:
        t1 = np.arange(len(spike))
        t2 = np.linspace(0, t1[-1], t1.size * 50)
        # Normalize
        spike /= np.max(np.abs(spike))
        sinterp = scipy.interpolate.CubicSpline(t1, spike)
        spike_ = sinterp(t2)
        spikes_pp.append(spike_)

    # We want to exclude neurons that do not have a peak following the trough        
    spike = np.mean(spikes_pp, axis=0)

    # Global minima
    gmin = np.argmin(spike)

    # FWHM of the depression
    fwhm = scipy.signal.peak_widths(-1*spike, [gmin])
    fwhm = fwhm[3] - fwhm[2]
    # local maxima
    pks = scipy.signal.find_peaks(spike, height=0, prominence=0.2)[0]

    # After the minimum
    pks = pks[pks > gmin]

    # But before too long...
    pks = pks[t2[pks] - t2[gmin] < 25]

    # Normalize
    if len(pks) == 0:
        return np.nan, np.nan, np.nan, np.nan
    else:
        # Prominence of the hyperpolarization peak above baseline
        baseline = np.mean(spike[0:500])
        hyp_height = spike[pks[0]] - baseline     

        # Using the averaged traces, the post peak inflection point may be measurable
        # Use just the first derivative to reduce numerical noise
        ds = np.diff(spike)
        ds = np.abs(ds)/np.max(np.abs(ds))
        
        dpks = scipy.signal.find_peaks(np.abs(ds)/np.max(np.abs(ds)), height=0.05)[0]
        if len(dpks > 0):
            dpks = dpks[dpks > pks[0]]
            if len(dpks > 0):
                return fwhm, pks[0] - gmin, dpks[0] - pks[0], hyp_height
            else:
                return np.nan, np.nan, np.nan, np.nan   
        else:
            return np.nan, np.nan, np.nan, np.nan


if __name__ == '__main__':
    regions = ['M1', 'S1']
        
    for region in regions:
        if region not in ['M1', 'S1']:
            raise NotImplementedError
    
        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        data_path = get_data_path(region)
        sessions = np.unique(df[session_key].values)

        if not os.path.exists(PATH_DICT['tmp'] + 'spike_wf_features_%s.pkl' % region):
            rl = []
            for i, session in tqdm(enumerate(sessions)):
                loader_args = df.iloc[0]['loader_args']
                loader_args['boxcox'] = None
                loader_args['high_pass'] = False
                loader_args['return_wf'] = True
                dat = load_data(data_path, region, session, loader_args)
                wf = dat['wf'].squeeze()
                spike_widths = []
                phrt = []
                fwhm = []
                hyp_height = []
                for cell in tqdm(wf):
                    fwhm_, sw, phrt_, hh = extract_wf_features(cell)
                    spike_widths.append(sw)
                    phrt.append(phrt_) 
                    fwhm.append(fwhm_)
                    hyp_height.append(hh)   

                r = {}
                r['session'] = session
                r['spike_widths'] = spike_widths
                r['phrt'] = phrt
                r['fwhm'] = fwhm
                r['hyp_height'] = hyp_height
                # Get the overall firing rate, and then the firing rate during the reach periods alone
                # Set boxcox=None to get raw spike counts
                assert(wf.shape[-1] == dat['spike_rates'].shape[-1])
                rl.append(r)


            with open(PATH_DICT['tmp'] + 'spike_wf_features_%s.pkl' % region, 'wb') as f:
                f.write(pickle.dumps(rl))

        else:
            with open(PATH_DICT['tmp'] + 'spike_wf_features_%s.pkl' % region, 'rb') as f:
                rl = pickle.load(f)
        
        spike_wf_df = pd.DataFrame(rl)
