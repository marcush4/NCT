import pdb
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
import itertools

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from region_select import *
from config import PATH_DICT

sys.path.append(PATH_DICT['repo'])
from utils import calc_loadings

from loaders import load_peanut
from decoders import lr_decoder

# Region specific plotting arguments
ylabels = {
    'M1': 'Velocity Prediction ' + r'$r^2$',
    'M1_trialized': 'Velocity Prediction ' + r'$r^2$',
    'S1': 'Velocity Prediction ' + r'$r^2$',
    'HPC': 'Position Prediction' + r'$r^2$',
    'M1_maze': 'Velocity Prediction' + r'$r^2$',
    'ML': 'Classification Accuracy',
    'AM': 'Classification Accuracy',
    'mPFC': 'Position Prediction' + r'$r^2$'
}

diff_yticks = {
    'M1': [0., 0.12],
    'M1_trialized': [0., 0.06],
    'S1': [0., 0.1],
    'HPC': [0, 0.20],
    'M1_maze':[0, 0.06],
    'ML': [0, 0.12],
    'AM': [0, 0.12],
    'mPFC': [0, 0.1]
}

diff_ylims = {
    'M1': [0, 0.125],
    'M1_trialized': [0, 0.06],
    'S1': [0., 0.11],
    'HPC': [0, 0.20],
    'M1_maze': [0, 0.06],
    'ML': [-0.042, 0.125],
    'AM': [-0.01, 0.125],
    'mPFC': [-0.1, 0.1]
}

xlim_dict = {
    'M1':[1, 30],
    'M1_trialized':[1, 30],
    'S1': [1, 30],
    'HPC': [1, 30],
    'M1_maze':[1, 30],
    'ML':[1, 59],
    'AM':[1, 59],
    'mPFC': [1, 30]
}

xtick_dict = {
    'M1':[1, 15, 30],
    'M1_trialized':[1, 15, 30],
    'S1':[1, 15, 30],
    'HPC':[1, 15, 30],
    'M1_maze':[1, 15, 30],
    'ML':[1, 25, 50],
    'AM':[1, 25, 50],
    'mPFC': [1, 15, 30]
}

ytick_dict = {
    'M1':[0., 0.2, 0.4],
    'M1_trialized':[0., 0.2, 0.4],
    'S1': [0., 0.25],
    'HPC': [0., 0.3, 0.6],
    'M1_maze': [0., 0.2, 0.4],
    'ML': [0., 0.35, 0.75],
    'AM': [0., 0.15, 0.3],
    'mPFC':[0, 0.25]
}

inset_locs = {
    'M1':[0.6, 0.1, 0.35, 0.35],
    'M1_trialized':[0.6, 0.1, 0.35, 0.35],
    'S1':[0.8, 0.1, 0.35, 0.35],
    'HPC':[0.6, 0.1, 0.35, 0.35],
    'M1_maze': [0.1, 0.65, 0.35, 0.35],
    'ML': [0.6, 0.1, 0.35, 0.35],
    'AM': [0.68, 0.08, 0.35, 0.35],
    'mPFC': [0.68, 0.08, 0.35, 0.35]
}

from region_select import loader_kwargs
def get_decoding_performance(df, region):
    if region in ['M1', 'S1', 'M1_trialized']:
        return df.iloc[0]['r2'][1]
    elif region == 'HPC':
        if np.isscalar(df.iloc[0]['r2']):   
            return df.iloc[0]['r2']
        else:
            return df.iloc[0]['r2'][0]
    elif region == 'M1_maze':
        return df.iloc[0]['r2'][1]
    elif region in ['ML', 'AM']:
        return 1 - df.iloc[0]['loss']
    elif region in ['mPFC']:
        return df.iloc[0]['r2'][0]

if __name__ == '__main__':

    #regions = ['M1', 'S1', 'M1_trialized', 'HPC', 'AM', 'ML']
    regions = ['M1', 'S1']

    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = PATH_DICT['figs']

    for region in regions:
        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        #df, session_key = load_decoding_df(region, **lkw)        
        sessions = np.unique(df[session_key].values)

        dims = np.unique(df['dim'].values)
        r2fc = np.zeros((len(sessions), dims.size, 5))
        r2pca = np.zeros((len(sessions), dims.size, 5))
        for i, session in tqdm(enumerate(sessions)):
            for j, dim in enumerate(dims):               
                for f in range(5):
                    df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':['LQGCA', 'FCCA']}
                    dim_fold_df = apply_df_filters(df, **df_filter)
                    assert(dim_fold_df.shape[0] == 1)
                    r2fc[i, j, f] = get_decoding_performance(dim_fold_df, region)
                    df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':'PCA'}
                    pca_df = apply_df_filters(df, **df_filter)
                    assert(pca_df.shape[0] == 1)
                    r2pca[i, j, f] = get_decoding_performance(pca_df, region)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        colors = ['black', 'red', '#781820', '#5563fa']
        dim_vals = dims
        n = len(sessions)
        # FCCA averaged over folds
        fca_r2 = np.mean(r2fc, axis=2)
        # PCA
        pca_r2 = np.mean(r2pca, axis=2)
        ax.fill_between(dim_vals, np.mean(fca_r2, axis=0) + np.std(fca_r2, axis=0)/np.sqrt(n),
                        np.mean(fca_r2, axis=0) - np.std(fca_r2, axis=0)/np.sqrt(n), color=colors[1], alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2, axis=0), color=colors[1])

        ax.fill_between(dim_vals, np.mean(pca_r2, axis=0) + np.std(pca_r2, axis=0)/np.sqrt(n),
                        np.mean(pca_r2, axis=0) - np.std(pca_r2, axis=0)/np.sqrt(n), color=colors[0], alpha=0.25)
        ax.plot(dim_vals, np.mean(pca_r2, axis=0), color=colors[0])
        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel(ylabels[region], fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xlim(xlim_dict[region])
        ax.set_xticks(xtick_dict[region])
        ax.set_yticks(ytick_dict[region])
        # Add legend manually
        # ax.legend(['FBC', 'FFC'], fontsize=10, loc='upper left', frameon=False)
        axin = ax.inset_axes(inset_locs[region])
        pca_auc = np.sum(pca_r2, axis=1)
        fca_auc = np.sum(fca_r2, axis=1)
        # Run a signed rank test
        _, p = scipy.stats.wilcoxon(pca_auc, fca_auc, alternative='less')
        print('Across session WCSRT: %f' % p)
        axin.scatter(np.zeros(n), pca_auc, color='k', alpha=0.75, s=3)
        axin.scatter(np.ones(n), fca_auc, color='r', alpha=0.75, s=3)
        axin.plot(np.array([(0, 1) for _ in range(pca_r2.shape[0])]).T, np.array([(y1, y2) for y1, y2 in zip(np.sum(pca_r2, axis=1), np.sum(fca_r2, axis=1))]).T, color='k', alpha=0.5)
        axin.set_yticks([])
        axin.set_ylabel('Decoding AUC', fontsize=10)
        axin.set_xlim([-0.5, 1.5])
        axin.set_xticks([0, 1])
        axin.set_xticklabels(['FFC', 'FBC'], fontsize=10)

        fig.savefig('%s/%s_decodingvdim.pdf' % (figpath, region), 
                    bbox_inches='tight', pad_inches=0)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # Plot of the differences across dimensions
        ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(n),
                        np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(n), color='blue', alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2 - pca_r2, axis=0), color='blue')
        max_delta = np.max(np.mean(fca_r2 - pca_r2, axis=0))
        fractional_delta = max_delta/np.mean(pca_r2, axis=0)[np.argmax(np.mean(fca_r2 - pca_r2, axis=0))]
        print('%s peak fractional improvement:%f' % (region, fractional_delta))

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel(r'$\Delta$' + ' ' + ylabels[region], fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        #ax.vlines(6, 0, np.mean(fca_r2 - pca_r2, axis=0)[5], linestyles='dashed', color='blue')
        #ax.hlines(np.mean(fca_r2 - pca_r2, axis=0)[5], 0, 6, linestyles='dashed', color='blue')
        ax.set_xlim([1, 30])
        ax.set_xticks([1, 6, 15, 30])
        ax.set_yticks(diff_yticks[region])
        ax.set_ylim(diff_ylims[region])

        fig.savefig('%s/%s_decoding_delta.pdf' % (figpath, region), bbox_inches='tight', pad_inches=0)

        # Summary statistics    
        # dr2 = np.divide(fca_r2 - pca_r2, pca_r2)
        # print('Mean Peak Fractional improvement: %f' % np.mean(np.max(dr2, axis=-1)))
        # # print('S.E. Fractional improvement: %f' % )
        # se = np.std(np.max(dr2, axis=-1))/np.sqrt(dr2.shape[0])
        # print('S.E. Peak Fractional improvement: %f' % se)

# med = np.median(np.max(dr2, axis=-1))
# print('Median Peak Fractional improvement: %f' % med)
# iqr25 = np.quantile(np.max(dr2, axis=-1), 0.25)
# iqr75 = np.quantile(np.max(dr2, axis=-1), 0.75)
# print('IQR Peak Fractional Improvement: (%f, %f)' % (iqr25, iqr75))


# delta_r2_auc = np.array([y2 - y1 for y1, y2 in zip(pca_auc, fca_auc)])
# print('Mean dAUC: %f' % np.mean(delta_r2_auc))
# print('S.E. dAUC: %f' % (np.std(delta_r2_auc)/np.sqrt(delta_r2_auc.size)))

# med = np.median(delta_r2_auc)
# print('Median dAUC: %f' % med)
# print('IQR dAUC: (%f, %f)' % (np.quantile(delta_r2_auc, 0.25),   
#                               np.quantile(delta_r2_auc, 0.75)))


