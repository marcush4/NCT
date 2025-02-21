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


sys.path.append('/home/akumar/nse/neural_control')
from utils import calc_loadings

from loaders import load_peanut
from decoders import lr_decoder

# Region specific plotting arguments
ylabels = {
    'M1': 'Velocity Prediction ' + r'$r^2$',
    'S1': 'Velocity Prediction ' + r'$r^2$',
    'HPC_peanut': 'Position Prediction' + r'$r^2$',
    'M1_maze': 'Position Prediction' + r'$r^2$',
    'ML': 'Classification Accuracy',
    'AM': 'Classification Accuracy'
}

diff_yticks = {
    'M1': [0., 0.12],
    'S1': [0., 0.1],
    'HPC_peanut': [0, 0.12],
    'M1_maze':[0, 0.12],
    'ML': [0, 0.12],
    'AM': [0, 0.12]
}

diff_ylims = {
    'M1': [0, 0.125],
    'S1': [0., 0.11],
    'HPC_peanut': [0, 0.125],
    'M1_maze': [0, 0.12],
    'ML': [-0.042, 0.125],
    'AM': [-0.01, 0.125]
}

xlim_dict = {
    'M1':[1, 30],
    'S1': [1, 30],
    'HPC_peanut': [1, 30],
    'M1_maze':[1, 30],
    'ML':[1, 59],
    'AM':[1, 59]
}

xtick_dict = {
    'M1':[1, 15, 30],
    'S1':[1, 15, 30],
    'HPC_peanut':[1, 15, 30],
    'M1_maze':[1, 15, 30],
    'ML':[1, 25, 50],
    'AM':[1, 25, 50]
}

ytick_dict = {
    'M1':[0., 0.2, 0.4],
    'S1': [0., 0.25],
    'HPC_peanut': [0., 0.2, 0.4],
    'M1_maze': [0., 0.2, 0.4],
    'ML': [0., 0.35, 0.75],
    'AM': [0., 0.15, 0.3]
}

inset_locs = {
    'M1':[0.6, 0.1, 0.35, 0.35],
    'S1':[0.8, 0.1, 0.35, 0.35],
    'HPC_peanut':[0.6, 0.1, 0.35, 0.35],
    'M1_maze': [0.6, 0.1, 0.35, 0.35],
    'ML': [0.6, 0.1, 0.35, 0.35],
    'AM': [0.6, 0.1, 0.35, 0.35]
}

from region_select import loader_kwargs
def get_decoding_performance(df, region):
    if region == 'M1':
        return df.iloc[0]['r2'][1]
    elif region == 'S1':
        return df.iloc[0]['r2'][1]
    elif region == 'HPC_peanut':
        return df.iloc[0]['r2']
    elif region == 'M1_maze':
        return df.iloc[0]['r2'][1]
    elif region in ['ML', 'AM']:
        return 1 - df.iloc[0]['loss']

if __name__ == '__main__':

    regions = ['M1_maze']
    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/ankit_kumar/neural_control/figs/mcmaze_smoothatdecode'

    for region in tqdm(regions):
        # root_path = '/clusterfs/NSDS_data/FCCA/postprocessed'
        root_path = '/home/ankit_kumar/Data/FCCA_revisions'
        with open(root_path + '/mcmaze_smoothatdecode_decoding_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df_master = pd.DataFrame(rl)
        data_files = np.unique(df_master['data_file'].values)
        # Subset of recording sessions
        data_files = data_files[[0, 1, 2, 3, 8]]
        df_master = apply_df_filters(df_master, data_file=list(data_files))

        # Filter the dimreduc loader args down to what works best without smoothing
        df_master = apply_df_filters(df_master, 
        loader_args = {
        'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':1,
        'trialize':True, 'interval':'after_go', 'trial_threshold':0.5
        } )

        loader_args = []
        sigma = np.array([0.5, 1, 2, 4])
        bw = [25]
        for b in bw:
            for s in sigma:
                loader_args.append({'bin_width':b, 'filter_fn':'gaussian', 'filter_kwargs':{'sigma':s}, 'boxcox':0.5, 
        'spike_threshold':1,
                                    'trialize':True, 'interval':'after_go', 'trial_threshold':0.5})
                loader_args.append({'bin_width':b, 'filter_fn':'gaussian', 'filter_kwargs':{'sigma':s}, 'boxcox':0.5, 
        'spike_threshold':1,
                                    'trialize':False})

        # bw = np.array([5, 10, 25, 50])
        # sigma = np.array([0.5, 1, 2, 4, 8])
        # for b in bw:
        #     for s in sigma:
        #         loader_args.append({'bin_width':b, 'filter_fn':'gaussian', 'filter_kwargs':{'sigma':s}, 'boxcox':0.5, 'spike_threshold':1,
        #                             'trialize':True, 'interval':'after_go', 'trial_threshold':0.5})
        #         loader_args.append({'bin_width':b, 'filter_fn':'gaussian', 'filter_kwargs':{'sigma':s}, 'boxcox':0.5, 'spike_threshold':1,
        #                             'trialize':False})
        dimreduc_args = [{'T':3, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42},
                         {'T':5, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42}]

        decoder_args = [
            {'trainlag': 3, 'testlag': 3, 'decoding_window':5},
            {'trainlag': 1, 'testlag': 1, 'decoding_window':5}            
        ]

        # bw = np.array([10, 25, 50])
        # for b in bw:
        #     loader_args.append({'bin_width':b, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':1,
        #                         'trialize':False})
        # for b in bw:
        #     loader_args.append({'bin_width':b, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':1,
        #                         'trialize':True, 'interval':'full', 'trial_threshold':0.5})
        #     loader_args.append({'bin_width':b, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':1,
        #                         'trialize':True, 'interval':'after_go', 'trial_threshold':0.5})

        # dimreduc_args = [{'T':3, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42},
        #                  {'T':2, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42},
        #                  {'T':1, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42},
        #                  {'T':5, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42}]

        # decoder_args = [{'trainlag': 3, 'testlag': 3, 'decoding_window':5},
        #                 {'trainlag': 1, 'testlag': 1, 'decoding_window':5},
        #                 {'trainlag': 2, 'testlag': 2, 'decoding_window':5}]
        # Try all combinations of loader, decoder and 
        loader_index = np.arange(len(loader_args))
        dimreduc_index = np.arange(len(dimreduc_args))
        decoder_index = np.arange(len(decoder_args))

        param_combs = itertools.product(loader_index, dimreduc_index, decoder_index)
        session_key = 'data_file'

        for loader_params in tqdm(param_combs):
            load_idx = loader_params[0]
            dr_idx = loader_params[1]
            dec_idx = loader_params[2]
            # loader args specify the decoder loader args
            df = apply_df_filters(df_master, dec_loader_args=loader_args[load_idx], decoder_args=decoder_args[dec_idx])
            # select FCCA hyperparameters and then re-merge
            df_pca = apply_df_filters(df, dimreduc_method='PCA')
            df_fcca = apply_df_filters(df, dimreduc_method='LQGCA', dimreduc_args=dimreduc_args[dr_idx])
            df = pd.concat([df_pca, df_fcca])
            if df.shape[0] == 0:
                continue
            # df, session_key = load_decoding_df(region, **loader_kwargs[region])
            #df, session_key = load_decoding_df(region, **lkw)
            session_key = 'data_file'
            
            sessions = np.unique(df[session_key].values)
            dims = np.unique(df['dim'].values)
            r2fc = np.zeros((len(sessions), dims.size, 5))
            r2pca = np.zeros((len(sessions), dims.size, 5))
            for i, session in tqdm(enumerate(sessions)):
                for j, dim in enumerate(dims):               
                    for f in range(5):
                        df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':['LQGCA', 'FCCA']}
                        dim_fold_df = apply_df_filters(df, **df_filter)
                        try:
                            assert(dim_fold_df.shape[0] == 1)
                            r2fc[i, j, f] = get_decoding_performance(dim_fold_df, region)
                        except:
                            r2fc[i, j, f] = np.nan
                        df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':'PCA'}
                        pca_df = apply_df_filters(df, **df_filter)
                        try:
                            assert(pca_df.shape[0] == 1)
                            r2pca[i, j, f] = get_decoding_performance(pca_df, region)
                        except:
                            r2pca[i, j, f] = np.nan
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

            colors = ['black', 'red', '#781820', '#5563fa']
            dim_vals = dims
            n = len(sessions)
            # FCCA averaged over folds
            fca_r2 = np.nanmean(r2fc, axis=2)
            # PCA
            pca_r2 = np.nanmean(r2pca, axis=2)
            ax.fill_between(dim_vals, np.nanmean(fca_r2, axis=0) + np.nanstd(fca_r2, axis=0)/np.sqrt(n),
                            np.nanmean(fca_r2, axis=0) - np.nanstd(fca_r2, axis=0)/np.sqrt(n), color=colors[1], alpha=0.25)
            ax.plot(dim_vals, np.nanmean(fca_r2, axis=0), color=colors[1])

            ax.fill_between(dim_vals, np.nanmean(pca_r2, axis=0) + np.nanstd(pca_r2, axis=0)/np.sqrt(n),
                            np.nanmean(pca_r2, axis=0) - np.nanstd(pca_r2, axis=0)/np.sqrt(n), color=colors[0], alpha=0.25)
            ax.plot(dim_vals, np.nanmean(pca_r2, axis=0), color=colors[0])
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
            #axin.set_title('****')
            #fig.tight_layout()
            fig.savefig('%s/%s_decodingvdim_vel_l%ddr%ddec%d.png' % 
            (figpath, region, loader_params[0], loader_params[1], loader_params[2]), 
            bbox_inches='tight', pad_inches=0)

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            # Plot of the differences across dimensions
            ax.fill_between(dim_vals, np.nanmean(fca_r2 - pca_r2, axis=0) + np.nanstd(fca_r2 - pca_r2, axis=0)/np.sqrt(n),
                            np.nanmean(fca_r2 - pca_r2, axis=0) - np.nanstd(fca_r2 - pca_r2, axis=0)/np.sqrt(n), color='blue', alpha=0.25)
            ax.plot(dim_vals, np.nanmean(fca_r2 - pca_r2, axis=0), color='blue')
            max_delta = np.max(np.nanmean(fca_r2 - pca_r2, axis=0))
            fractional_delta = max_delta/np.nanmean(pca_r2, axis=0)[np.argmax(np.nanmean(fca_r2 - pca_r2, axis=0))]
            print('%s peak fractional improvement:%f' % (region, fractional_delta))

            ax.set_xlabel('Dimension', fontsize=18)
            ax.set_ylabel(r'$\Delta$' + ' ' + ylabels[region], fontsize=18)
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)

            #ax.vlines(6, 0, np.nanmean(fca_r2 - pca_r2, axis=0)[5], linestyles='dashed', color='blue')
            #ax.hlines(np.nanmean(fca_r2 - pca_r2, axis=0)[5], 0, 6, linestyles='dashed', color='blue')
            ax.set_xlim([1, 30])
            ax.set_xticks([1, 6, 15, 30])
            ax.set_yticks(diff_yticks[region])
            ax.set_ylim(diff_ylims[region])

            fig.savefig('%s/%s_decoding_delta_vel_l%ddr%ddec%d.png' % 
            (figpath, region, loader_params[0], loader_params[1], loader_params[2]), bbox_inches='tight', pad_inches=0)

        # Summary statistics    
        # dr2 = np.divide(fca_r2 - pca_r2, pca_r2)
        # print('Mean Peak Fractional improvement: %f' % np.nanmean(np.max(dr2, axis=-1)))
        # # print('S.E. Fractional improvement: %f' % )
        # se = np.nanstd(np.max(dr2, axis=-1))/np.sqrt(dr2.shape[0])
        # print('S.E. Peak Fractional improvement: %f' % se)

        # med = np.median(np.max(dr2, axis=-1))
        # print('Median Peak Fractional improvement: %f' % med)
        # iqr25 = np.quantile(np.max(dr2, axis=-1), 0.25)
        # iqr75 = np.quantile(np.max(dr2, axis=-1), 0.75)
        # print('IQR Peak Fractional Improvement: (%f, %f)' % (iqr25, iqr75))


        # delta_r2_auc = np.array([y2 - y1 for y1, y2 in zip(pca_auc, fca_auc)])
        # print('Mean dAUC: %f' % np.nanmean(delta_r2_auc))
        # print('S.E. dAUC: %f' % (np.nanstd(delta_r2_auc)/np.sqrt(delta_r2_auc.size)))

        # med = np.median(delta_r2_auc)
        # print('Median dAUC: %f' % med)
        # print('IQR dAUC: (%f, %f)' % (np.quantile(delta_r2_auc, 0.25),   
        #                               np.quantile(delta_r2_auc, 0.75)))


