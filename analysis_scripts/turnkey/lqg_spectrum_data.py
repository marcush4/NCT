import pdb
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
import itertools

import torch
from sklearn.model_selection import KFold
from config import PATH_DICT
from region_select import *

sys.path.append(PATH_DICT['repo'])
from utils import apply_df_filters
from FCCA.fcca import lqg_spectrum
from FCCA.fcca import LQGComponentsAnalysis as FCCA

regions = ['M1', 'S1', 'HPC_peanut', 'AM', 'ML']
# regions = ['HPC_peanut']

# Dimension to use for combined plot
DIM_DICT = {
    'M1': 30,
    'S1': 30,
    'HPC_peanut': 30,
    'AM': 30,
    'ML':30,
}


def calc_lqg_spect(df, region):
    # What is the LQG T?
    df_lqg = apply_df_filters(df, dimreduc_method=['LQGCA', 'FCCA'])
    T = df_lqg.iloc[0]['dimreduc_args']['T']

    data_path = get_data_path(region)

    sessions = np.unique(df[session_key].values)
    dims = np.unique(df['dim'].values)
    folds = np.unique(df['fold_idx'].values)


    spect_fcca = np.zeros((sessions.size, dims.size, folds.size), dtype=object)
    spect_pca = np.zeros((sessions.size, dims.size, folds.size), dtype=object)

    for i, session in tqdm(enumerate(sessions)):
        # Load data
        if 'full_arg_tuple' in df.keys():
            # Modify the full_arg_tuple according to the desired loader args
            df_sess = apply_df_filters(df, **{session_key:session, 'loader_args':{'region': region}})
            full_arg_tuple = [df_sess.iloc[0]['full_arg_tuple']]
        else:
            full_arg_tuple = None
        
        dat = load_data(data_path, region, session,
                        df.iloc[0]['loader_args'], full_arg_tuple)
        X = dat['spike_rates'].squeeze()
        for j, dim in enumerate(dims):  
            for k, (train_idxs, test_idxs) in \
            enumerate(KFold(folds.size).split(np.arange(X.shape[0]))):
                Xtr = X[train_idxs]   
                # ccm = calc_cross_cov_mats_from_data(X)
                lqgmodel = FCCA(T=T)
                lqgmodel.estimate_data_statistics(Xtr)
                
                # Calculate the spectrum given projections
                df_filter = {session_key:session, 'dim':dim,
                            'fold_idx': k, 'dimreduc_method':['LQGCA', 'FCCA']}
                df_f = apply_df_filters(df, **df_filter)
                assert(df_f.shape[0] == 1)
                df_filter = {session_key:session, 'dim':dim,
                            'fold_idx': k, 'dimreduc_method':'PCA'}
                df_p = apply_df_filters(df, **df_filter)
                assert(df_p.shape[0] == 1)
                Vfcca = df_f.iloc[0]['coef']
                Vpca = df_p.iloc[0]['coef'][:, 0:dim]
                spec1 = lqg_spectrum(torch.tensor(Vfcca), 
                                    lqgmodel.cross_covs, lqgmodel.cross_covs_rev)
                spect_fcca[i, j, k] = spec1
                # Asssert LQG spectrum from PCA
                spec2 = lqg_spectrum(torch.tensor(Vpca), 
                                    lqgmodel.cross_covs, lqgmodel.cross_covs_rev)
                spect_pca[i, j, k] = spec2
    with open(PATH_DICT['tmp'] + '/lqg_spect_%s.pkl' % region, 'wb') as f:
        f.write(pickle.dumps(spect_fcca))
        f.write(pickle.dumps(spect_pca))


def all_plots(region, df):

    sessions = np.unique(df[session_key].values)
    dims = np.unique(df['dim'].values)
    folds = np.unique(df['fold_idx'].values)

    with open(PATH_DICT['tmp'] + '/lqg_spect_%s.pkl' % region, 'rb') as f:
        spect_fcca = pickle.load(f)
        spect_pca = pickle.load(f)

    if not os.path.exists('lqg_spectrum'):
        os.makedirs('lqg_spectrum')

    if not os.path.exists('lqg_spectrum/%s' % region):
        os.makedirs('lqg_spectrum/%s' % region)

    # Plot across dimensions and sessions
    for i, session in tqdm(enumerate(sessions)):
        for j, dim in enumerate(dims):
            fig, ax = plt.subplots(figsize=(4, 4))
            # Plot the normalized cumsum
            z1 = np.max(spect_fcca[i, j, 0])
            #y = np.cumsum(np.sort(spect_fcca[i, j, 0]))
            y1 = np.sort(spect_fcca[i, j, 0])
            #ax.plot(y/z, color='r')

            z2 = np.max(spect_pca[i, j, 0])
            #y = np.cumsum(np.sort(spect_fcca[i, j, 0]))
            y2 = np.sort(spect_pca[i, j, 0])
            #ax.plot(y/z, color='k')
            dy = y2 - y1
            dy /= np.max(dy)
            ax.plot(y2 - y1)
            # ax.plot(np.sort(spect_fcca[i, j, 0]), color='r')
            # ax.plot(np.sort(spect_pca[i, j, 0]), color='k')
            fig.savefig(f'lqg_spectrum/{region}/{i}_{j}.png')
            plt.close(fig)
        sys.exit()

def summary_plot():
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    if not os.path.exists(PATH_DICT['figs'] + '/lqg_spectrum'):
        os.makedirs(PATH_DICT['figs'] + '/lqg_spectrum')

    for region in regions:
        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        sessions = np.unique(df[session_key].values)
        dims = np.unique(df['dim'].values)
        # Fix this
        dim_index = np.argmin(np.abs(dims - DIM_DICT[region]))
        folds = np.unique(df['fold_idx'].values)
        with open(PATH_DICT['tmp'] + '/lqg_spect_%s.pkl' % region, 'rb') as f:
            spect_fcca = pickle.load(f)
            spect_pca = pickle.load(f)

        # Select the index according to DIM_DICT, average over
        # sessions and folds
        y1 = spect_fcca[:, dim_index, :].ravel()
        # Sort each element
        yorder = [np.argsort(y) for y in y1]
        y1 = np.array([np.sort(y) for y in y1])
        # Truncate to sdim dimensions. If too short, pad with nans
        sdim = 200
        # Used for ploting
        sdim2 = 100
        for i, y in enumerate(y1):
            if len(y) > sdim:
                y = y[:sdim]
            y = np.pad(y, (0, sdim - len(y)), constant_values=np.nan)  
            y = np.ma.masked_array(y, mask=np.isnan(y))                  
            y1[i] = y

        y1 = np.ma.concatenate([y[np.newaxis, :] for y in y1])

        y2 = spect_pca[:, dim_index, :].ravel()
        y2 = np.array([np.sort(y) for y in y2])
        # y2 = np.array([y[yorder[i]] for i, y in enumerate(y2)])
        for i, y in enumerate(y2):
            if len(y) > sdim:
                y = y[:sdim]
            y = np.pad(y, (0, sdim - len(y)), constant_values=np.nan)                    
            y = np.ma.masked_array(y, mask=np.isnan(y))                  
            y2[i] = y

        y2 = np.ma.concatenate([y[np.newaxis, :] for y in y2])
        dy = y2 - y1
        # dy = np.cumsum(y2, axis=-1) - np.cumsum(y1, axis=-1)
        dy = np.divide(dy, np.nanmax(dy, axis=1, keepdims=True))
        # Sort so that NaNs end up at the start, and then we reverse order
        dy.sort(axis=1, endwith=False)
        dy = dy[:, ::-1]
        dyavg = np.nanmean(dy, axis=0)[0:sdim2]
        dystd = np.nanstd(dy, axis=0)[0:sdim2]/np.sqrt(dy.shape[0])
        ax.plot(np.arange(1, sdim2 + 1), dyavg)
        ax.fill_between(np.arange(1, sdim2 + 1), dyavg - dystd, dyavg + dystd, alpha=0.25)                


    ax.legend(['M1', 'S1', 'HPC', 'AM', 'ML'])
    ax.set_xticks([1, 25, 50, 75, 100])
    ax.set_yticks([0, 0.5, 1.])
    fig.savefig('lqg_spect_summary.png')

if __name__ == '__main__':

    # calculate spectrum
    # for region in regions:
    #     df, session_key = load_decoding_df(region, **loader_kwargs[region])
    #     if not os.path.exists(PATH_DICT['tmp'] + '/lqg_spect_%s.pkl' % region):
    #         calc_lqg_spect(region)
    #     all_plots(region, df)
    summary_plot()