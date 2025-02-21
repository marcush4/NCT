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
    'M1_psid': 'Velocity Prediction ' + r'$r^2$',
    'M1_trialized': 'Velocity Prediction ' + r'$r^2$',
    'S1': 'Velocity Prediction ' + r'$r^2$',
    'S1_psid': 'Velocity Prediction ' + r'$r^2$',
    'HPC_peanut': 'Position Prediction' + r'$r^2$',
    'HPC': 'Position Prediction' + r'$r^2$',
    'M1_maze': 'Velocity Prediction' + r'$r^2$',
    'ML': 'Classification Accuracy',
    'AM': 'Classification Accuracy',
    'mPFC': 'Position Prediction' + r'$r^2$',
    'VISp': 'Classification Accuracy'
}

diff_yticks = {
    'M1': [0., 0.12],
    'M1_psid': [0., 0.12],
    'M1_trialized': [0., 0.06],
    'S1': [0., 0.1],
    'S1_psid': [0., 0.1],
    'HPC_peanut': [0, 0.25],
    'HPC': [0, 0.20],
    'M1_maze':[0, 0.06],
    'ML': [0, 0.12],
    'AM': [0, 0.12],
    'mPFC': [0, 0.1],
    'VISp': [0, 0.12],
}

diff_ylims = {
    'M1': [0, 0.125],
    'M1_psid': [0, 0.13],
    'M1_trialized': [0, 0.06],
    'S1': [0., 0.11],
    'S1_psid': [0., 0.11],
    'HPC_peanut': [0, 0.25],
    'HPC': [0, 0.20],
    'M1_maze': [0, 0.06],
    'ML': [-0.042, 0.125],
    'AM': [-0.01, 0.125],
    'mPFC': [-0.1, 0.1],
    'VISp': [-0.1, 0.1]
}

xlim_dict = {
    'M1':[1, 30],
    'M1_psid':[1, 30],
    'M1_trialized':[1, 30],
    'S1': [1, 30],
    'S1_psid': [1, 30],
    'HPC_peanut': [1, 30],
    'HPC': [1, 30],
    'M1_maze':[1, 30],
    'ML':[1, 59],
    'AM':[1, 59],
    'mPFC': [1, 30],
    'VISp':[1,30]
}

xtick_dict = {
    'M1':[1, 15, 30],
    'M1_psid':[1, 15, 30],
    'M1_trialized':[1, 15, 30],
    'S1':[1, 15, 30],
    'S1_psid':[1, 15, 30],
    'HPC_peanut':[1, 15, 30],
    'HPC':[1, 15, 30],
    'M1_maze':[1, 15, 30],
    'ML':[1, 25, 50],
    'AM':[1, 25, 50],
    'mPFC': [1, 15, 30],
    'VISp': [1, 15, 30]
}

ytick_dict = {
    'M1':[0., 0.2, 0.4],
    'M1_psid':[0., 0.25, 0.5],
    'M1_trialized':[0., 0.2, 0.4],
    'S1': [0., 0.25],
    'S1_psid': [0., 0.25],
    'HPC_peanut': [0., 0.3, 0.6],
    'HPC': [0., 0.3, 0.6],
    'M1_maze': [0., 0.2, 0.4],
    'ML': [0., 0.4, 0.8], #[0., 0.35, 0.75],
    'AM': [0., 0.4, 0.8], #[0., 0.15, 0.3],
    'mPFC':[0, 0.25],
    'VISp': [0, 0.25]
}

inset_locs = {
    'M1':[0.6, 0.1, 0.35, 0.35],
    'M1_psid':[0.8, 0.1, 0.35, 0.35],
    'M1_trialized':[0.6, 0.1, 0.35, 0.35],
    'S1':[0.8, 0.1, 0.35, 0.35],
    'S1_psid':[0.8, 0.1, 0.35, 0.35],
    'HPC_peanut':[0.6, 0.1, 0.35, 0.35],
    'HPC':[0.6, 0.1, 0.35, 0.35],
    'M1_maze': [0.1, 0.65, 0.35, 0.35],
    'ML': [0.6, 0.1, 0.35, 0.35],
    'AM': [0.68, 0.08, 0.35, 0.35],
    'mPFC': [0.68, 0.08, 0.35, 0.35],
    'VISp': [0.68, 0.08, 0.35, 0.35]

}

from region_select import loader_kwargs
def get_decoding_performance(df, region, **kwargs):
    if region in ['M1', 'S1', 'M1_trialized', 'M1_psid', 'S1_psid']:
        return df.iloc[0]['r2'][1]
    elif region in ['M1_psid_rand', 'S1_psid_rand']:
        return np.array([df.iloc[k]['r2'][1] for k in range(df.shape[0])])
    elif region in ['M1_psid_sup', 'S1_psid_sup']:
        return df.iloc[0]['r2'][1][kwargs['dim_index']]
    elif region == 'HPC_peanut':
        if np.isscalar(df.iloc[0]['r2']):   
            return df.iloc[0]['r2']
        else:
            return df.iloc[0]['r2'][0]
    elif region == 'HPC_peanut_rand':
        return np.array([df.iloc[k]['r2'][0] for k in range(df.shape[0])])
    elif region == 'HPC_peanut_sup':
        # Return the asymptotic (i.e. rank 2) position performance
        return df.iloc[0]['r2'][0][-1]
    elif region == 'M1_maze':
        return df.iloc[0]['r2'][1]
    elif region in ['ML', 'AM']:
        return 1 - df.iloc[0]['loss']
    elif region in ['ML_sup', 'AM_sup']:
        return 1 - df.iloc[0]['r2'][0]['loss']
    elif region in ['ML_rand', 'AM_rand']:
        pdb.set_trace()
    elif region in ['mPFC', 'HPC']:
        return df.iloc[0]['r2'][0]
    elif region in ['VISp']:
        return 1 - df.iloc[0]['loss']
    else:
        raise NotImplementedError


if __name__ == '__main__':

    #regions = ['M1', 'S1', 'M1_trialized', 'HPC', 'AM', 'ML']
    # regions = ['mPFC']
    # regions = ['ML', 'AM']
    # regions = ['M1_psid', 'S1_psid']
    regions = ['VISp']
    # regions = ['M1_trialized', 'M1_maze', 'M1_psid']

    include_rand_control = False
    nrand = 5 #1000
    include_supervised_ub = False
    
    Recompute_Controls = True
    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = PATH_DICT['figs']

    r2p_across_regions = []
    r2f_across_regions = []
    sessions_per_region = []

                
        

        
        
    for region in regions:
        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        
        sessions = np.unique(df[session_key].values)
        sessions_per_region.append(sessions)
        dims = np.unique(df['dim'].values)
        folds = np.unique(df['fold_idx'].values)
        r2fc = np.zeros((len(sessions), dims.size, folds.size))
        r2pca = np.zeros((len(sessions), dims.size, folds.size))
    
        if not Recompute_Controls:
            if include_rand_control:
                df_rand = load_rand_decoding_df(region, **loader_kwargs[region])
                dims_rand = np.unique(df_rand['dim'].values)
            if include_supervised_ub:
                df_sup, dim_key = load_supervised_decoding_df(region, **loader_kwargs[region])
                dims_sup = df_sup.iloc[0]['decoder_args'][dim_key]
                
        else:

            if include_rand_control:
                if not os.path.exists(PATH_DICT['tmp'] + '/rand_decoding_%s_rand.pkl' % region):
                    compute_rand = True
                    dims_rand = dims
                    r2_rand = np.zeros((len(sessions), dims_rand.size, folds.size, nrand))
                else:
                    compute_rand = False
                    with open(PATH_DICT['tmp'] + '/rand_decoding_%s_rand.pkl' % region, 'rb') as f:
                        r2_rand = pickle.load(f)
            else:
                compute_rand = False
            
            if include_supervised_ub:
                dims_sup = dims
                r2_sup = np.zeros((len(sessions), dims_sup.size, folds.size))
                
            for i, session in tqdm(enumerate(sessions)):
                for j, dim in enumerate(dims):               
                    for f in folds:
                        df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':['LQGCA', 'FCCA']}
                        dim_fold_df = apply_df_filters(df, **df_filter)
                        try:
                            assert(dim_fold_df.shape[0] == 1)
                        except:
                            pdb.set_trace()
                        r2fc[i, j, f] = get_decoding_performance(dim_fold_df, region)
                        df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 
                                    'dimreduc_method':'PCA'}
                        pca_df = apply_df_filters(df, **df_filter)
                        assert(pca_df.shape[0] == 1)
                        r2pca[i, j, f] = get_decoding_performance(pca_df, region)

                if include_supervised_ub:
                    for f in folds:
                        df_filter = {session_key:session, 'fold_idx':f}
                        sup_df = apply_df_filters(df_sup, **df_filter)
                        for j, dim in enumerate(dims_sup):
                            r2_sup[i, j, f] = get_decoding_performance(sup_df, region + '_sup', dim_index=j)

                if include_rand_control and compute_rand:
                    for j, dim in enumerate(dims_rand):
                        for f in folds:
                            df_filter = {'dim':dim, session_key:session, 'fold_idx':f}
                            rand_df = apply_df_filters(df, **df_filter)
                            # Perhaps this dim value was not included
                            if rand_df.shape[0] == 0:
                                r2_rand[i, j, f] = np.nan
                            else:
                                r2_rand[i, j, f] = get_decoding_performance(rand_df, region + '_rand')
            # Rand assemblage takes time so save it as tmp
            if compute_rand:
                with open(PATH_DICT['tmp'] + '/rand_decoding_%s_rand.pkl' % region, 'wb') as f:
                    f.write(pickle.dumps(r2_rand))

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        colors = ['black', 'red', '#781820', '#8a4be3']
        dim_vals = dims
        n = len(sessions)
        # FCCA averaged over folds
        fca_r2 = np.mean(r2fc, axis=2)
        # PCA
        pca_r2 = np.mean(r2pca, axis=2)

        print(np.mean(fca_r2, axis=0))
        print(np.mean(pca_r2, axis=0))

        ax.fill_between(dim_vals, np.mean(fca_r2, axis=0) + np.std(fca_r2, axis=0)/np.sqrt(n),
                        np.mean(fca_r2, axis=0) - np.std(fca_r2, axis=0)/np.sqrt(n), color=colors[1], alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2, axis=0), color=colors[1])

        ax.fill_between(dim_vals, np.mean(pca_r2, axis=0) + np.std(pca_r2, axis=0)/np.sqrt(n),
                        np.mean(pca_r2, axis=0) - np.std(pca_r2, axis=0)/np.sqrt(n), color=colors[0], alpha=0.25)
        ax.plot(dim_vals, np.mean(pca_r2, axis=0), color=colors[0])

        if include_rand_control:
            # Average over folds and nrand, and then show the std err across sessions
            r2_rand_ = np.nanmean(r2_rand, axis=(-1, -2))
            yavg = np.mean(r2_rand_, axis=0)
            ystd = np.std(r2_rand_, axis=0)
            ax.fill_between(dims_rand, yavg + ystd/np.sqrt(n),
                            yavg - ystd/np.sqrt(n), 
                            color=colors[2], alpha=0.25)
            ax.plot(dims_rand, yavg, color=colors[2])

        if include_supervised_ub:
            # Average over folds 
            r2_sup_ = np.nanmean(r2_sup, axis=2)
            yavg = np.mean(r2_sup_, axis=0)
            ystd = np.std(r2_sup_, axis=0)
            ax.fill_between(dims_sup, yavg + ystd/np.sqrt(n),
                            yavg - ystd/np.sqrt(n), 
                            color=colors[3], alpha=0.25)
            ax.plot(dims_sup, yavg, color=colors[3])


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

        fractional_delta = np.divide(np.mean(fca_r2 - pca_r2, axis=0),
                                     np.mean(pca_r2, axis=0))    
        max_delta = np.argmax(np.mean(fca_r2 - pca_r2, axis=0))
        #pdb.set_trace()
        peak_fractional_improvement = fractional_delta[max_delta]

        print('%s peak fractional improvement:%f' % (region, peak_fractional_improvement))

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel(r'$\Delta$' + ' ' + ylabels[region], fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        #ax.vlines(6, 0, np.mean(fca_r2 - pca_r2, axis=0)[5], linestyles='dashed', color='blue')
        #ax.hlines(np.mean(fca_r2 - pca_r2, axis=0)[5], 0, 6, linestyles='dashed', color='blue')
        ax.set_xlim([1, 30])
        ax.set_xticks([1, 15, 30])
        ax.set_yticks(diff_yticks[region])
        ax.set_ylim(diff_ylims[region])

        fig.savefig('%s/%s_decoding_delta.pdf' % (figpath, region), bbox_inches='tight', pad_inches=0)

        r2f_across_regions.append(fca_r2)
        r2p_across_regions.append(pca_r2)

        # Summary statistics    
        # dr2 = np.divide(fca_r2 - pca_r2, pca_r2)
        # print('Mean Peak Fractional improvement: %f' % np.mean(np.max(dr2, axis=-1)))
        # # print('S.E. Fractional improvement: %f' % )
        # se = np.std(np.max(dr2, axis=-1))/np.sqrt(dr2.shape[0])
        # print('S.E. Peak Fractional improvement: %f' % se)

    if len(regions) == 2:
        # Summary statistics region comparison            
        _, p1 = scipy.stats.mannwhitneyu(np.sum(r2f_across_regions[0], axis=1, keepdims=True),
                                         np.sum(r2f_across_regions[1], axis=1, keepdims=True),
                                         alternative='greater', axis=0)

        _, p2 = scipy.stats.mannwhitneyu(np.sum(r2p_across_regions[0], axis=1, keepdims=True),
                                         np.sum(r2p_across_regions[1], axis=1, keepdims=True),
                                         alternative='greater', axis=0)

    else:

        try:
            # Comparison between M1 and its trialized version
            m1_trialized_idx = regions.index('M1_trialized')
            m1_idx = regions.index('M1_psid')
            m1_maze_idx = regions.index('M1_maze')
        except ValueError:
            sys.exit()

        


        # M1 trialized is missing a session...
        common_session_indices = [i for i, elem in enumerate(sessions_per_region[m1_idx])
                                  if elem in sessions_per_region[m1_trialized_idx]]
        
        common_session_indices = np.array(common_session_indices)

        # First comparison - difference between M1/M1 trialized at d=30
        fr1 = r2f_across_regions[m1_idx][common_session_indices]
        fr2 = r2f_across_regions[m1_trialized_idx]
        fr3 = r2f_across_regions[m1_maze_idx]

        pr1 = r2p_across_regions[m1_idx][common_session_indices]
        pr2 = r2p_across_regions[m1_trialized_idx]
        pr3 = r2f_across_regions[m1_maze_idx]

        _, p1 = scipy.stats.wilcoxon(fr1[:, -1], fr2[:, -1], alternative='greater')
        _, p2 = scipy.stats.wilcoxon(pr1[:, -1], pr2[:, -1], alternative='greater')

        print(f'FBC M1 vs. M1 trialized p = {p1}, FFC p = {p2}')

        # Second comparison - difference between difference in FBC/FFC at d=6
        dr1 = fr1 - pr1
        dr2 = fr2 - pr2

        _, p3 = scipy.stats.wilcoxon(dr1, dr2, alternative='greater')
        print(f'Delta decoding p = {p3}')
