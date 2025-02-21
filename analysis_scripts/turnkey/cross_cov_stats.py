import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import sys

from npeet.entropy_estimators import entropy as knn_entropy
from dca.cov_util import calc_cross_cov_mats_from_data, calc_pi_from_cross_cov_mats

from region_select import *
from config import PATH_DICT
from Fig4 import get_loadings_df

sys.path.append(PATH_DICT['repo'])
from utils import apply_df_filters, calc_loadings
from loaders import load_sabes
from segmentation import reach_segment_sabes


def get_cross_cov_stats(loadings_df, 
                        session_key, data_path, region,
                        quantile=0.75):

    data_path = globals()['data_path']
    sessions = np.unique(loadings_df[session_key].values)

    pk_cross_covs = np.zeros((len(sessions), 2))
    cross_cov_entropy = np.zeros((len(sessions), 2))

    for h, session in enumerate(sessions):
        x, time = get_rates_smoothed(data_path, region, session, return_t=True, std=True,   
                                     trial_average=True, sigma=2)        
        T = time.size

        # Get loadings that exceed quantile
        df_ = apply_df_filters(loadings_df, **{session_key:session})
        loadings_fcca = df_['FCCA_loadings'].values
        loadings_pca = df_['PCA_loadings'].values
        assert(x.shape[0] == df_.shape[0])
        # quantile cutoff
        loadings_fcca_cutoff = np.quantile(loadings_fcca, quantile)
        loadings_pca_cutoff = np.quantile(loadings_pca, quantile)

        # Get indices of loadings that exceed quantile
        loadings_fcca_idx = np.argwhere(loadings_fcca > loadings_fcca_cutoff).squeeze()
        loadings_pca_idx = np.argwhere(loadings_pca > loadings_pca_cutoff).squeeze()

        # Get top neurons
        x_f = x[loadings_fcca_idx]
        x_p = x[loadings_pca_idx]
        n = int((x_f.shape[0]**2 - x_f.shape[0])/2)
        cross_covs_f = np.zeros((n, T))
        cross_covs_p = np.zeros((n, T))

        flat_idx = 0
        for j in range(x_f.shape[0]):
            for k in range(j, x_f.shape[0]):
                if j == k:
                    continue
                cc = np.correlate(x_f[j], x_f[k], mode='same')/T
                cross_covs_f[flat_idx] = cc                
                flat_idx += 1
        
        flat_idx = 0
        for j in range(x_p.shape[0]):
            for k in range(j, x_p.shape[0]):
                if j == k:
                    continue
                cc = np.correlate(x_p[j], x_p[k], mode='same')/T
                cross_covs_p[flat_idx] = cc                
                flat_idx += 1


        pk_cross_covs[h, 0] = np.mean(np.max(cross_covs_f, axis=1))
        pk_cross_covs[h, 1] = np.mean(np.max(cross_covs_p, axis=1))

        tau_max_f = np.argmax(cross_covs_f, axis=1)
        tau_max_p = np.argmax(cross_covs_p, axis=1)

        cross_cov_entropy[h, 0] = knn_entropy(tau_max_f[:, np.newaxis])
        cross_cov_entropy[h, 1] = knn_entropy(tau_max_p[:, np.newaxis])

    return pk_cross_covs, cross_cov_entropy

dim_dict = {
    'M1': 6,
    'S1': 6,
    'M1_trialized':6,
    'HPC_peanut': 11,
    'M1_maze':6,
    'AM': 21,
    'ML': 21,
    'mPFC': 5
}

if __name__ == '__main__':
    regions = ['M1', 'S1', 'HPC_peanut']

    pvals = []
    for region in regions:
        data_path = get_data_path(region)
        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        loadings_df = get_loadings_df(df, session_key, dim_dict[region])
        pk_cross_covs, cross_cov_entropy = get_cross_cov_stats(loadings_df, 
                                                               session_key, data_path, region)
        # Paired difference tests
        wstat1, p1 = scipy.stats.wilcoxon(pk_cross_covs[:, 0], pk_cross_covs[:, 1], alternative='less')
        wstat2, p2 = scipy.stats.wilcoxon(cross_cov_entropy[:, 0], cross_cov_entropy[:, 1], alternative='greater')
        pvals.append((p1, p2))
    for i, region in enumerate(regions):
        p1, p2 = pvals[i]
        print(f'{region}:')
        print(f'pk_cross_covs: p={p1}')
        print(f'cross_cov_entropy: p={p2}')
