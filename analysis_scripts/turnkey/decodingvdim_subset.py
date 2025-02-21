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

from utils import apply_df_filters, calc_loadings

from loaders import load_peanut
from decoders import lr_decoder

if __name__ == '__main__':

    M1 = False
    figpath = PATH_DICT['figs']

    if M1:
        # Co-plot original decoding results with subset selected decoding results
        df_original, session_key = load_decoding_df('M1')
        
        with open(PATH_DICT['df'] + '/sabes_subset_dense_decoding_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df_subset = pd.DataFrame(rl)

        # taken from the submit file
        subset_path = '/home/ankit_kumar/neural_control/subset_indices/'
        subset_files = list(glob.glob('%spca_subset_*_M1.pkl' % subset_path))
        subset_files.extend(glob.glob('%sfca_subset_*_M1.pkl' % subset_path))

    else:
        df_original, session_key = load_decoding_df('S1')

        with open(PATH_DICT['df'] + '/sabes_subset_dense_decodingS1_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df_subset = pd.DataFrame(rl)

        # taken from the submit file
        subset_path = '/home/ankit_kumar/neural_control/subset_indices/'
        subset_files = list(glob.glob('%spca_subset_*_S1.pkl' % subset_path))
        subset_files.extend(glob.glob('%sfca_subset_*_S1.pkl' % subset_path))

    # Subset files
    data_files = np.unique(df_original['data_file'].values)
    quantiles = np.unique([int(sfile.split('_q')[1].split('_')[0])
                           for sfile in subset_files])
    dims = np.unique(df_subset['dim'].values)

    if M1:
        tmp_path = PATH_DICT['tmp'] + '/M1subset_dense_decoding_tmp.pkl'
    else:
        tmp_path = PATH_DICT['tmp'] + '/S1subset_dense_decoding_tmp.pkl'
    if not os.path.exists(tmp_path):
        r2fc = np.zeros((len(data_files), dims.size, 5))
        r2pca = np.zeros((len(data_files), dims.size, 5))

        r2fc_ss = np.zeros((len(data_files), dims.size, 5, len(subset_files)))
        r2pca_ss = np.zeros((len(data_files), dims.size, 5, len(subset_files)))

        for i, data_file in tqdm(enumerate(data_files)):
            for j, dim in enumerate(dims):               
                for f in range(5):
                    dim_fold_df = apply_df_filters(df_original, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='LQGCA')
                    assert(dim_fold_df.shape[0] == 1)
                    r2fc[i, j, f] = dim_fold_df.iloc[0]['r2'][1]
                    pca_df = apply_df_filters(df_original, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='PCA')
                    assert(pca_df.shape[0] == 1)
                    r2pca[i, j, f] = pca_df.iloc[0]['r2'][1]

                    dim_fold_subset = apply_df_filters(df_subset, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='LQGCA')
                    assert(dim_fold_subset.shape[0] == len(subset_files))
                    for k, sfile in enumerate(subset_files):
                        idx = [kk for kk in range(len(subset_files))
                            if dim_fold_subset.iloc[kk]['loader_args']['subset_file'] == sfile][0]
                        d_ = dim_fold_subset.iloc[idx]
                        r2fc_ss[i, j, f, k] = d_['r2'][1]

                    dim_fold_subset = apply_df_filters(df_subset, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='PCA')
                    assert(dim_fold_subset.shape[0] == len(subset_files))
                    for k, sfile in enumerate(subset_files):
                        idx = [kk for kk in range(len(subset_files))
                            if dim_fold_subset.iloc[kk]['loader_args']['subset_file'] == sfile][0]
                        d_ = dim_fold_subset.iloc[idx]
                        r2pca_ss[i, j, f, k] = d_['r2'][1]
        if M1:
            with open(tmp_path, 'wb') as f:
                f.write(pickle.dumps(r2fc))
                f.write(pickle.dumps(r2pca))
                f.write(pickle.dumps(r2fc_ss))
                f.write(pickle.dumps(r2pca_ss))
        else:
            with open(tmp_path, 'wb') as f:
                f.write(pickle.dumps(r2fc))
                f.write(pickle.dumps(r2pca))
                f.write(pickle.dumps(r2fc_ss))
                f.write(pickle.dumps(r2pca_ss))

    else:
        if M1:
            with open(tmp_path, 'rb') as f:
                r2fc = pickle.load(f)
                r2pca = pickle.load(f)
                r2fc_ss = pickle.load(f)
                r2pca_ss = pickle.load(f)            
        else:
            with open(tmp_path, 'rb') as f:
                r2fc = pickle.load(f)
                r2pca = pickle.load(f)
                r2fc_ss = pickle.load(f)
                r2pca_ss = pickle.load(f)            

    # Group subset files by quantile
    subset_labels = ['rFFC excl.', 'rFBC excl.']
    for q in quantiles:
        
        if M1:
            sfile_idx1 = subset_files.index('%spca_subset_q%d_M1.pkl' % (subset_path, q))
            sfile_idx2 = subset_files.index('%sfca_subset_q%d_M1.pkl' % (subset_path, q))
        else:
            sfile_idx1 = subset_files.index('%spca_subset_q%d_S1.pkl' % (subset_path, q))
            sfile_idx2 = subset_files.index('%sfca_subset_q%d_S1.pkl' % (subset_path, q))

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        colors = ['black', 'red', '#781820', '#5563fa']
        dim_vals = dims

        # FCCA averaged over folds
        fca_r2_og = np.mean(r2fc, axis=2)
        # PCA
        pca_r2_og = np.mean(r2pca, axis=2)
        #ax.fill_between(dim_vals, np.mean(fca_r2, axis=0) + np.std(fca_r2, axis=0)/np.sqrt(35),
        #                np.mean(fca_r2, axis=0) - np.std(fca_r2, axis=0)/np.sqrt(35), color=colors[1], alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2_og, axis=0), color=colors[1])

        #ax.fill_between(dim_vals, np.mean(pca_r2, axis=0) + np.std(pca_r2, axis=0)/np.sqrt(35),
        #                np.mean(pca_r2, axis=0) - np.std(pca_r2, axis=0)/np.sqrt(35), color=colors[0], alpha=0.25)
        ax.plot(dim_vals, np.mean(pca_r2_og, axis=0), color=colors[0])

        linestyles = ['None', 'None', 'None', 'None']
        markerstyles = ['s', 'v', '1', '*']
        for k, sidx in enumerate([sfile_idx1, sfile_idx2]):
            fca_r2 = np.mean(r2fc_ss[..., sidx], axis=2)
            # PCA
            pca_r2 = np.mean(r2pca_ss[..., sidx], axis=2)
            #ax.fill_between(dim_vals, np.mean(fca_r2, axis=0) + np.std(fca_r2, axis=0)/np.sqrt(35),
            #                np.mean(fca_r2, axis=0) - np.std(fca_r2, axis=0)/np.sqrt(35), color=colors[1], alpha=0.25)
            ax.plot(dim_vals, np.mean(fca_r2, axis=0), color=colors[1],
                    linestyle=linestyles[k], marker=markerstyles[k], alpha=0.25)

            #ax.fill_between(dim_vals, np.mean(pca_r2, axis=0) + np.std(pca_r2, axis=0)/np.sqrt(35),
            #                np.mean(pca_r2, axis=0) - np.std(pca_r2, axis=0)/np.sqrt(35), color=colors[0], alpha=0.25)
            ax.plot(dim_vals, np.mean(pca_r2, axis=0), color=colors[0],
                    linestyle=linestyles[k], marker=markerstyles[k], alpha=0.25)

        # Plot the paired differences

        # ax.plot(dim_vals, )
        # ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35),
        #                 np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35), color=colors[0], alpha=0.25)

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel('Velocity Prediction ' + r'$r^2$', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xlim([1, 30])
        ax.set_xticks([1, 15, 30])
        ax.set_yticks([0., 0.2, 0.4])

        #ax.legend(['FCCA', 'PCA'], fontsize=10, bbox_to_anchor=(0.32, 1.01), frameon=False)
        labels = ['FBC', 'FFC']
        for lbl in subset_labels:
            labels.extend(['FBC %s' % lbl, 'FFC %s' % lbl])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(labels, fontsize=10, loc='lower right', frameon=False,
                    bbox_to_anchor=(1.7, 0.0))
        ax.set_title('Exclusion quantile %d' % q)
        if M1:
            fig.savefig('%s/indy_vel_decoding_subset_q%d.pdf' % (figpath, q), bbox_inches='tight', pad_inches=0)
        else:
            fig.savefig('%s/indy_vel_decoding_subsetS1_q%d.pdf' % (figpath, q), bbox_inches='tight', pad_inches=0)

    # Summarize across quantiles - plot the AUC in paired differences

    fig, ax = plt.subplots(figsize=(8, 4))
    fca_diffs_ffc_excl = []
    fca_diffs_fbc_excl = []

    pca_diffs_ffc_excl = []
    pca_diffs_fbc_excl = []
    for q in quantiles:
        if M1:
            sfile_idx1 = subset_files.index('%spca_subset_q%d_M1.pkl' % (subset_path, q))
            sfile_idx2 = subset_files.index('%sfca_subset_q%d_M1.pkl' % (subset_path, q))
        else:
            sfile_idx1 = subset_files.index('%spca_subset_q%d_S1.pkl' % (subset_path, q))
            sfile_idx2 = subset_files.index('%sfca_subset_q%d_S1.pkl' % (subset_path, q))

        fca_diffs_ffc_excl.append(r2fc - r2fc_ss[..., sfile_idx1])
        fca_diffs_fbc_excl.append(r2fc - r2fc_ss[..., sfile_idx2])

        pca_diffs_ffc_excl.append(r2pca - r2pca_ss[..., sfile_idx1])
        pca_diffs_fbc_excl.append(r2pca - r2pca_ss[..., sfile_idx2])

    fca_diffs_ffc_excl = np.array(fca_diffs_ffc_excl)
    fca_diffs_fbc_excl = np.array(fca_diffs_fbc_excl)
    pca_diffs_ffc_excl = np.array(pca_diffs_ffc_excl)
    pca_diffs_fbc_excl = np.array(pca_diffs_fbc_excl)

    # Take the delta at d=6
    didx = list(dims).index(6)
    # fca_diffs_ffc_excl = np.sum(fca_diffs_ffc_excl, axis=2)
    fca_diffs_ffc_excl = fca_diffs_ffc_excl[..., didx, :]
    fca_diffs_ffc_excl = np.reshape(fca_diffs_ffc_excl, (fca_diffs_ffc_excl.shape[0], -1))

    # fca_diffs_fbc_excl = np.sum(fca_diffs_fbc_excl, axis=2)
    fca_diffs_fbc_excl = fca_diffs_fbc_excl[..., didx, :]
    fca_diffs_fbc_excl = np.reshape(fca_diffs_fbc_excl, (fca_diffs_fbc_excl.shape[0], -1))

    # pca_diffs_ffc_excl = np.sum(pca_diffs_ffc_excl, axis=2)
    pca_diffs_ffc_excl = pca_diffs_ffc_excl[..., didx, :]
    pca_diffs_ffc_excl = np.reshape(pca_diffs_ffc_excl, (pca_diffs_ffc_excl.shape[0], -1))

    # pca_diffs_fbc_excl = np.sum(pca_diffs_fbc_excl, axis=2)
    pca_diffs_fbc_excl = pca_diffs_fbc_excl[..., didx, :]
    pca_diffs_fbc_excl = np.reshape(pca_diffs_fbc_excl, (pca_diffs_fbc_excl.shape[0], -1))

    # Stack together
    y = np.vstack([fca_diffs_ffc_excl[0:1], fca_diffs_fbc_excl[0:1], 
                    pca_diffs_ffc_excl[0:1], pca_diffs_fbc_excl[0:1]])

    # For a sense of scale, let's calculate the delta decoding between FCCA and PCA at d=6
    dr2 = r2fc - r2pca
    dr2 = dr2[:, didx, :]
    dr2 = np.mean(dr2)

    for i in range(1, len(quantiles)):
        y = np.vstack([y, fca_diffs_ffc_excl[i-1:i], fca_diffs_fbc_excl[i-1:i], 
                        pca_diffs_ffc_excl[i-1:i], pca_diffs_fbc_excl[i-1:i]])


    delta = 2
    positions = np.concatenate([np.arange(4*k + 2*k, 4*(k+1) + 2*k) for k in range(len(quantiles))])
    bplot = ax.boxplot(y.T, positions=positions, patch_artist=True)

    ax.set_xticks([1.5 + 6*k for k in range(8)])
    ax.set_xticklabels(['Q. %d' % q for q in quantiles])
    ax.set_ylabel(r'$\Delta-r^2$')
    if M1:
        ax.set_ylim([0, 0.4])
    else:
        ax.set_ylim([0, 0.25])
    # Set box face colors to distinguish cases
    colors = ['#bf6969', 'r', 'k', '#9c8c8c']
    mediancolors = ['k', 'r', 'k', 'r']

    for i, median in enumerate(bplot['medians']):
        median.set_color(mediancolors[i % 4])
    for i, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(colors[i % 4])

    ax.legend([bplot['boxes'][0], bplot['boxes'][1], bplot['boxes'][2], bplot['boxes'][3]],
              ['FBC-FFC excl.', 'FBC-FBC excl.', 'FFC-FFC excl.', 'FFC-FBC excl.'], loc='upper right')

    ax.hlines(dr2, 0, 45, color='b')
    fig.tight_layout()

    if M1:
        fig.savefig('%s/M1_dense_subset_diffs_summ.pdf' % figpath, bbox_inches='tight', pad_inches=0)
    else:
        fig.savefig('%s/S1_dense_subset_diffs_summ.pdf' % figpath, bbox_inches='tight', pad_inches=0)
