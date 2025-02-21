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

from region_select import load_decoding_df


#sys.path.append('/home/akumar/nse/neural_control')
sys.path.append('/home/marcush/projects/neural_control/')
sys.path.append('/home/marcush/projects/neural_control/analysis_scripts/')
sys.path.append('/home/marcush/projects/github_repos')
from utils import apply_df_filters, calc_loadings

from loaders import load_peanut
from decoders import lr_decoder

if __name__ == '__main__':

    M1 = False
    S1 = False
    HPC = False
    AM = True
    ML = True

    # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        #figpath = '/home/akumar/nse/neural_control/figs/revisions'
        figpath = '/home/marcush/projects/neural_control/notebooks/TsaoLabNotebooks/CodeAndFigsForPaper/Figs'

    if M1:

        df, _ = load_decoding_df('M1')

        data_files = np.unique(df['data_file'].values)
        dims = np.unique(df['dim'].values)

   
        r2fc = np.zeros((len(data_files), dims.size, 5))
        r2pca = np.zeros((len(data_files), dims.size, 5))

        for i, data_file in tqdm(enumerate(data_files)):
            for j, dim in enumerate(dims):               
                for f in range(5):
                    dim_fold_df = apply_df_filters(df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='LQGCA')
                    assert(dim_fold_df.shape[0] == 1)
                    r2fc[i, j, f] = dim_fold_df.iloc[0]['r2'][1]
                    pca_df = apply_df_filters(df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='PCA')
                    assert(pca_df.shape[0] == 1)
                    r2pca[i, j, f] = pca_df.iloc[0]['r2'][1]


        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        # Average across folds and plot
        # REINSERT OLS(5) IN HERE IF NEEDED

        colors = ['black', 'red', '#781820', '#5563fa']
        dim_vals = dims

        # # DCA averaged over folds
        # dca_r2 = np.mean(r2[:, :, 1, :, 1], axis=2)
        # # KCA averaged over folds
        # kca_r2 = np.mean(r2[:, :, 2, :, 1], axis=2)

        # FCCA averaged over folds
        fca_r2 = np.mean(r2fc[:, :, :], axis=2)
        # PCA
        pca_r2 = np.mean(r2pca, axis=2)
        # ax.fill_between(dim_vals, np.mean(dca_r2, axis=0) + np.std(dca_r2, axis=0)/np.sqrt(35),
        #                 np.mean(dca_r2, axis=0) - np.std(dca_r2, axis=0)/np.sqrt(35), color=colors[0], alpha=0.25)
        # ax.plot(dim_vals, np.mean(dca_r2, axis=0), color=colors[0])
        # ax.fill_between(dim_vals, np.mean(kca_r2, axis=0) + np.std(kca_r2, axis=0)/np.sqrt(35),
        #                 np.mean(kca_r2, axis=0) - np.std(kca_r2, axis=0)/np.sqrt(35), color=colors[1], alpha=0.25)
        # ax.plot(dim_vals, np.mean(kca_r2, axis=0), color=colors[1])
        ax.fill_between(dim_vals, np.mean(fca_r2, axis=0) + np.std(fca_r2, axis=0)/np.sqrt(35),
                        np.mean(fca_r2, axis=0) - np.std(fca_r2, axis=0)/np.sqrt(35), color=colors[1], alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2, axis=0), color=colors[1])

        ax.fill_between(dim_vals, np.mean(pca_r2, axis=0) + np.std(pca_r2, axis=0)/np.sqrt(35),
                        np.mean(pca_r2, axis=0) - np.std(pca_r2, axis=0)/np.sqrt(35), color=colors[0], alpha=0.25)
        ax.plot(dim_vals, np.mean(pca_r2, axis=0), color=colors[0])

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
        ax.legend(['FBC', 'FFC'], fontsize=10, loc='upper left', frameon=False)

        #ax.legend(['FCCA', 'PCA'], fontsize=14, loc='lower right', frameon=False)
        
        #ax.set_title('Macaque M1', fontsize=18)

        # Inset that shows the paired differences
        #axin = ax.inset_axes([0.125, -0.924, 0.75, 0.75])
        axin = ax.inset_axes([0.6, 0.1, 0.35, 0.35])

        pca_auc = np.sum(pca_r2, axis=1)
        fca_auc = np.sum(fca_r2, axis=1)

        # Run a signed rank test
        _, p = scipy.stats.wilcoxon(pca_auc, fca_auc, alternative='less')
        print(p)

        axin.scatter(np.zeros(35), pca_auc, color='k', alpha=0.75, s=3)
        axin.scatter(np.ones(35), fca_auc, color='r', alpha=0.75, s=3)
        axin.plot(np.array([(0, 1) for _ in range(pca_r2.shape[0])]).T, np.array([(y1, y2) for y1, y2 in zip(np.sum(pca_r2, axis=1), np.sum(fca_r2, axis=1))]).T, color='k', alpha=0.5)
        axin.set_yticks([])
        axin.set_ylabel('Decoding AUC', fontsize=10)
        axin.set_xlim([-0.5, 1.5])
        axin.set_xticks([0, 1])
        axin.set_xticklabels(['FFC', 'FBC'], fontsize=10)
        #axin.set_title('****')
        #fig.tight_layout()
        fig.savefig('%s/indy_vel_decoding.pdf' % figpath, bbox_inches='tight', pad_inches=0)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # Plot of the differences across dimensions
        ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35),
                        np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35), color='blue', alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2 - pca_r2, axis=0), color='blue')

        max_delta = np.max(np.mean(fca_r2 - pca_r2, axis=0))
        fractional_delta = max_delta/np.mean(pca_r2, axis=0)[np.argmax(np.mean(fca_r2 - pca_r2, axis=0))]

        print('M1 peak fractional improvement:%f' % fractional_delta)

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel(r'$\Delta$' + ' Velocity Prediction ' + r'$r^2$', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        #ax.vlines(6, 0, np.mean(fca_r2 - pca_r2, axis=0)[5], linestyles='dashed', color='blue')
        #ax.hlines(np.mean(fca_r2 - pca_r2, axis=0)[5], 0, 6, linestyles='dashed', color='blue')
        ax.set_xlim([1, 30])
        ax.set_xticks([1, 6, 15, 30])
        ax.set_yticks([0., 0.12])
        ax.set_ylim([0, 0.125])

        fig.savefig('%s/indy_vel_decoding_delta.pdf' % figpath, bbox_inches='tight', pad_inches=0)


    if S1:
        df, _ = load_decoding_df('S1')
        data_files = np.unique(df['data_file'].values)
        dims = np.unique(df['dim'].values)



        r2fc = np.zeros((len(data_files), dims.size, 5))
        r2pca = np.zeros((len(data_files), dims.size, 5))

        for i, data_file in tqdm(enumerate(data_files)):
            for j, dim in enumerate(dims):               
                for f in range(5):
                    dim_fold_df = apply_df_filters(df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='LQGCA')
                    r2fc[i, j, f] = dim_fold_df.iloc[0]['r2'][1]
                    pca_df = apply_df_filters(df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='PCA')
                    assert(pca_df.shape[0] == 1)
                    r2pca[i, j, f] = pca_df.iloc[0]['r2'][1]

        # FCCA averaged over folds
        fca_r2 = np.mean(r2fc, axis=2)
        # PCA
        pca_r2 = np.mean(r2pca, axis=2)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        colors = ['black', 'red', '#781820', '#5563fa']
        dim_vals = dims

        ax.fill_between(dim_vals, np.mean(fca_r2, axis=0) + np.std(fca_r2, axis=0)/np.sqrt(8),
                        np.mean(fca_r2, axis=0) - np.std(fca_r2, axis=0)/np.sqrt(8), color=colors[1], alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2, axis=0), color=colors[1])

        ax.fill_between(dim_vals, np.mean(pca_r2, axis=0) + np.std(pca_r2, axis=0)/np.sqrt(8),
                        np.mean(pca_r2, axis=0) - np.std(pca_r2, axis=0)/np.sqrt(8), color=colors[0], alpha=0.25)
        ax.plot(dim_vals, np.mean(pca_r2, axis=0), color=colors[0])

        # Plot the paired differences

        # ax.plot(dim_vals, )
        # ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35),
        #                 np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35), color=colors[0], alpha=0.25)

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel('Velocity Prediction ' + r'$r^2$', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.legend(['FBC', 'FFC'], fontsize=10, bbox_to_anchor=(0.32, 1.01), frameon=False)
        ax.set_yticks([0., 0.25])
        ax.set_xticks([1., 15, 30])
        #ax.legend(['FCCA', 'PCA'], fontsize=14, loc='lower right', frameon=False)
        
        #ax.set_title('Macaque S1', fontsize=18)

        # Inset that shows the paired differences
        #axin = ax.inset_axes([0.125, -0.924, 0.75, 0.75])
        axin = ax.inset_axes([0.8, 0.1, 0.35, 0.35])

        pca_auc = np.sum(pca_r2, axis=1)
        fca_auc = np.sum(fca_r2, axis=1)

        # Run a signed rank test
        _, p = scipy.stats.wilcoxon(pca_auc, fca_auc, alternative='less')
        print(p)

        axin.scatter(np.zeros(8), pca_auc, color='k', alpha=0.75, s=3)
        axin.scatter(np.ones(8), fca_auc, color='r', alpha=0.75, s=3)
        axin.plot(np.array([(0, 1) for _ in range(pca_r2.shape[0])]).T, np.array([(y1, y2) for y1, y2 in zip(np.sum(pca_r2, axis=1), np.sum(fca_r2, axis=1))]).T, color='k', alpha=0.5)
        axin.set_yticks([])
        axin.set_ylabel('Decoding AUC', fontsize=10)
        axin.set_xlim([-0.5, 1.5])
        axin.set_xticks([0, 1])
        axin.set_xticklabels(['FFC', 'FBC'], fontsize=10)
        #axin.set_title('***')
        #fig.tight_layout()
        fig.savefig('%s/S1_vel_decoding.pdf' % figpath, bbox_inches='tight', pad_inches=0)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # Plot of the differences across dimensions
        ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(8),
                        np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(8), color='blue', alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2 - pca_r2, axis=0), color='blue')

        max_delta = np.max(np.mean(fca_r2 - pca_r2, axis=0))
        fractional_delta = max_delta/np.mean(pca_r2, axis=0)[np.argmax(np.mean(fca_r2 - pca_r2, axis=0))]

        print('S1 peak fractional improvement:%f' % fractional_delta)

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel(r'$\Delta$' + ' Velocity Prediction ' + r'$r^2$', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.set_yticks([0., 0.1])
        ax.set_ylim([0., 0.11])
        ax.set_xticks([1, 6, 15, 30])

        #ax.vlines(6, 0, np.mean(fca_r2 - pca_r2, axis=0)[5], linestyles='dashed', color='blue')
        #ax.hlines(np.mean(fca_r2 - pca_r2, axis=0)[5], 0, 6, linestyles='dashed', color='blue')
        ax.set_xlim([0, 30])
        fig.savefig('%s/S1_vel_decoding_delta.pdf' % figpath, bbox_inches='tight', pad_inches=0)
    if HPC:
        df, _ = load_decoding_df('HPC')

        epochs = np.unique(df['epoch'].values)
        dims = np.unique(df['dim'].values)

   
        r2fc = np.zeros((len(epochs), dims.size, 5))
        r2pca = np.zeros((len(epochs), dims.size, 5))

        for i, epoch in tqdm(enumerate(epochs)):
            for j, dim in enumerate(dims):               
                for f in range(5):
                    dim_fold_df = apply_df_filters(df, epoch=epoch, dim=dim, fold_idx=f, dimreduc_method='FCCA')
                    assert(dim_fold_df.shape[0] == 1)
                    r2fc[i, j, f] = dim_fold_df.iloc[0]['r2']
                    pca_df = apply_df_filters(df, epoch=epoch, dim=dim, fold_idx=f, dimreduc_method='PCA')
                    assert(pca_df.shape[0] == 1)
                    r2pca[i, j, f] = pca_df.iloc[0]['r2']

        pdb.set_trace()
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        # Average across folds and plot
        # REINSERT OLS(5) IN HERE IF NEEDED

        colors = ['black', 'red', '#781820', '#5563fa']
        dim_vals = dims

        # # DCA averaged over folds
        # dca_r2 = np.mean(r2[:, :, 1, :, 1], axis=2)
        # # KCA averaged over folds
        # kca_r2 = np.mean(r2[:, :, 2, :, 1], axis=2)

        # FCCA averaged over folds
        fca_r2 = np.mean(r2fc[:, :, :], axis=2)
        # PCA
        pca_r2 = np.mean(r2pca, axis=2)
        # ax.fill_between(dim_vals, np.mean(dca_r2, axis=0) + np.std(dca_r2, axis=0)/np.sqrt(35),
        #                 np.mean(dca_r2, axis=0) - np.std(dca_r2, axis=0)/np.sqrt(35), color=colors[0], alpha=0.25)
        # ax.plot(dim_vals, np.mean(dca_r2, axis=0), color=colors[0])
        # ax.fill_between(dim_vals, np.mean(kca_r2, axis=0) + np.std(kca_r2, axis=0)/np.sqrt(35),
        #                 np.mean(kca_r2, axis=0) - np.std(kca_r2, axis=0)/np.sqrt(35), color=colors[1], alpha=0.25)
        # ax.plot(dim_vals, np.mean(kca_r2, axis=0), color=colors[1])
        ax.fill_between(dim_vals, np.mean(fca_r2, axis=0) + np.std(fca_r2, axis=0)/np.sqrt(8),
                        np.mean(fca_r2, axis=0) - np.std(fca_r2, axis=0)/np.sqrt(8), color=colors[1], alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2, axis=0), color=colors[1])

        ax.fill_between(dim_vals, np.mean(pca_r2, axis=0) + np.std(pca_r2, axis=0)/np.sqrt(8),
                        np.mean(pca_r2, axis=0) - np.std(pca_r2, axis=0)/np.sqrt(8), color=colors[0], alpha=0.25)
        ax.plot(dim_vals, np.mean(pca_r2, axis=0), color=colors[0])

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
        ax.legend(['FBC', 'FFC'], fontsize=10, loc='upper left', frameon=False)

        #ax.legend(['FCCA', 'PCA'], fontsize=14, loc='lower right', frameon=False)
        
        #ax.set_title('Macaque M1', fontsize=18)

        # Inset that shows the paired differences
        #axin = ax.inset_axes([0.125, -0.924, 0.75, 0.75])
        axin = ax.inset_axes([0.6, 0.1, 0.35, 0.35])

        pca_auc = np.sum(pca_r2, axis=1)
        fca_auc = np.sum(fca_r2, axis=1)

        # Run a signed rank test
        _, p = scipy.stats.wilcoxon(pca_auc, fca_auc, alternative='less')
        print(p)

        axin.scatter(np.zeros(8), pca_auc, color='k', alpha=0.75, s=3)
        axin.scatter(np.ones(8), fca_auc, color='r', alpha=0.75, s=3)
        axin.plot(np.array([(0, 1) for _ in range(pca_r2.shape[0])]).T, np.array([(y1, y2) for y1, y2 in zip(np.sum(pca_r2, axis=1), np.sum(fca_r2, axis=1))]).T, color='k', alpha=0.5)
        axin.set_yticks([])
        axin.set_ylabel('Decoding AUC', fontsize=10)
        axin.set_xlim([-0.5, 1.5])
        axin.set_xticks([0, 1])
        axin.set_xticklabels(['FFC', 'FBC'], fontsize=10)
        #axin.set_title('****')
        #fig.tight_layout()
        fig.savefig('%s/peanut_decoding.pdf' % figpath, bbox_inches='tight', pad_inches=0)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # Plot of the differences across dimensions
        ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(8),
                        np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(8), color='blue', alpha=0.25)
        ax.plot(dim_vals, np.mean(fca_r2 - pca_r2, axis=0), color='blue')

        max_delta = np.max(np.mean(fca_r2 - pca_r2, axis=0))
        fractional_delta = max_delta/np.mean(pca_r2, axis=0)[np.argmax(np.mean(fca_r2 - pca_r2, axis=0))]

        print('HPC peak fractional improvement:%f' % fractional_delta)

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel(r'$\Delta$' + ' Velocity Prediction ' + r'$r^2$', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        #ax.vlines(6, 0, np.mean(fca_r2 - pca_r2, axis=0)[5], linestyles='dashed', color='blue')
        #ax.hlines(np.mean(fca_r2 - pca_r2, axis=0)[5], 0, 6, linestyles='dashed', color='blue')
        ax.set_xlim([1, 30])
        ax.set_xticks([1, 6, 15, 30])
        ax.set_yticks([0., 0.12])
        ax.set_ylim([0, 0.125])
        fig.savefig('%s/peanut_decoding_delta.pdf' % figpath, bbox_inches='tight', pad_inches=0)



    if 'ML':

        region = 'ML'
        manual_dim = 21
        df, session_key = load_decoding_df(region)

        ############ Getting Decoding Performance (1-Hamming Loss using logistic regression)
        data_files = np.unique(df['data_file'].values)
        dims = np.unique(df['dim'].values)
        nFolds = len(np.unique(df['fold_idx']))

        FFC_decoding_mat = np.zeros((data_files.size, dims.size, nFolds))
        FBC_decoding_mat = np.zeros((data_files.size, dims.size, nFolds))

        for i, data_file in tqdm(enumerate(data_files)):
            for j, dim in enumerate(dims):               
                for f in range(nFolds):
                    fbc_df = apply_df_filters(df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='LQGCA')
                    FBC_decoding_mat[i, j, f] = 1 - fbc_df.iloc[0]['loss']

                    ffc_df = apply_df_filters(df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='PCA')
                    FFC_decoding_mat[i, j, f] = 1 - ffc_df.iloc[0]['loss']

        fbc_dec = np.mean(FBC_decoding_mat, axis=2)
        ffc_dec = np.mean(FFC_decoding_mat, axis=2)


        ############ Plot 1 Code
        norm_factor = 1 # for the fold differences 
        colors = ['black', 'red', '#781820', '#5563fa']

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # FBC 
        ax.fill_between(dims, np.mean(fbc_dec, axis=0) + np.std(fbc_dec, axis=0)/np.sqrt(norm_factor), 
                        np.mean(fbc_dec, axis=0) - np.std(fbc_dec, axis=0)/np.sqrt(norm_factor), color=colors[1], alpha=0.25, label='_nolegend_')
        ax.plot(dims, np.mean(fbc_dec, axis=0), color=colors[1])

        # FFC
        ax.fill_between(dims, np.mean(ffc_dec, axis=0) + np.std(ffc_dec, axis=0)/np.sqrt(norm_factor),
                        np.mean(ffc_dec, axis=0) - np.std(ffc_dec, axis=0)/np.sqrt(norm_factor), color=colors[0], alpha=0.25, label='_nolegend_')
        ax.plot(dims, np.mean(ffc_dec, axis=0), color=colors[0])

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel('Classification Accuracy', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xlim([1, max(dims)])
        ax.set_xticks([1, int(max(dims)/2), int(max(dims))])
        ax.set_yticks([0., 0.35, 0.75])
        ax.legend(['FBC', 'FFC'], fontsize=10, loc='upper left', frameon=False)

       # Inset Plot
        pca_auc = np.sum(ffc_dec, axis=1)
        fca_auc = np.sum(fbc_dec, axis=1)
        _, p = scipy.stats.wilcoxon(pca_auc, fca_auc, alternative='less')

        axin = ax.inset_axes([0.6, 0.1, 0.35, 0.35])
        axin.scatter(np.zeros(len(pca_auc)), pca_auc, color='k', alpha=0.75, s=3)
        axin.scatter(np.ones(len(fca_auc)), fca_auc, color='r', alpha=0.75, s=3)
        axin.plot(np.array([(0, 1) for _ in range(ffc_dec.shape[0])]).T, np.array([(y1, y2) for y1, y2 in zip(np.sum(ffc_dec, axis=1), np.sum(fbc_dec, axis=1))]).T, color='k', alpha=0.5)
        axin.set_yticks([])
        axin.set_ylabel('Decoding AUC', fontsize=10)
        axin.set_xlim([-0.5, 1.5])
        axin.set_xticks([0, 1])
        axin.set_xticklabels(['FFC', 'FBC'], fontsize=10)

        fig.savefig('%s/Tsao_DecodingVDim_%s.pdf' % (figpath, region), bbox_inches='tight', pad_inches=0)

        ############# Main Plot 2
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        # Plot of the differences across dimensions
        ax.fill_between(dims, np.mean(fbc_dec - ffc_dec, axis=0) + np.std(fbc_dec - ffc_dec, axis=0)/np.sqrt(35),
                        np.mean(fbc_dec - ffc_dec, axis=0) - np.std(fbc_dec - ffc_dec, axis=0)/np.sqrt(35), color='blue', alpha=0.25)
        ax.plot(dims, np.mean(fbc_dec - ffc_dec, axis=0), color='blue')

        max_delta = np.max(np.mean(fbc_dec - ffc_dec, axis=0))
        fractional_delta = max_delta/np.mean(ffc_dec, axis=0)[np.argmax(np.mean(fbc_dec - ffc_dec, axis=0))]
        #print('Peak fractional improvement:%f' % fractional_delta)

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel(r'$\Delta$' + ' Classification Accuracy ', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.vlines(manual_dim, np.min(np.mean(fbc_dec - ffc_dec, axis=0)), np.mean(fbc_dec - ffc_dec, axis=0)[dims == manual_dim], linestyles='dashed', color='blue')
        ax.hlines(np.mean(fbc_dec - ffc_dec, axis=0)[dims == manual_dim], 0, manual_dim, linestyles='dashed', color='blue')
        ax.set_xlim([1, 30])
        ax.set_xticks([1, manual_dim, int(max(dims)/2), int(max(dims))])
        ax.set_yticks([0., 0.12])
        ax.set_ylim([-0.042, 0.125])

        fig.savefig('%s/Tsao_DecodingVDim_Delta_%s.pdf' % (figpath, region), bbox_inches='tight', pad_inches=0)



    if 'AM':

        region = 'AM'
        manual_dim = 21
        df, session_key = load_decoding_df(region)

        ############ Getting Decoding Performance (1-Hamming Loss using logistic regression)
        data_files = np.unique(df['data_file'].values)
        dims = np.unique(df['dim'].values)
        nFolds = len(np.unique(df['fold_idx']))

        FFC_decoding_mat = np.zeros((data_files.size, dims.size, nFolds))
        FBC_decoding_mat = np.zeros((data_files.size, dims.size, nFolds))

        for i, data_file in tqdm(enumerate(data_files)):
            for j, dim in enumerate(dims):               
                for f in range(nFolds):
                    fbc_df = apply_df_filters(df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='LQGCA')
                    FBC_decoding_mat[i, j, f] = 1 - fbc_df.iloc[0]['loss']

                    ffc_df = apply_df_filters(df, data_file=data_file, dim=dim, fold_idx=f, dimreduc_method='PCA')
                    FFC_decoding_mat[i, j, f] = 1 - ffc_df.iloc[0]['loss']

        fbc_dec = np.mean(FBC_decoding_mat, axis=2)
        ffc_dec = np.mean(FFC_decoding_mat, axis=2)


        ################################################ Plot Code
        ############# Main Plot 1
        norm_factor = 1 # for the fold differences 
        colors = ['black', 'red', '#781820', '#5563fa']
 
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # FBC 
        ax.fill_between(dims, np.mean(fbc_dec, axis=0) + np.std(fbc_dec, axis=0)/np.sqrt(norm_factor), 
                        np.mean(fbc_dec, axis=0) - np.std(fbc_dec, axis=0)/np.sqrt(norm_factor), color=colors[1], alpha=0.25, label='_nolegend_')
        ax.plot(dims, np.mean(fbc_dec, axis=0), color=colors[1])

        # FFC
        ax.fill_between(dims, np.mean(ffc_dec, axis=0) + np.std(ffc_dec, axis=0)/np.sqrt(norm_factor),
                        np.mean(ffc_dec, axis=0) - np.std(ffc_dec, axis=0)/np.sqrt(norm_factor), color=colors[0], alpha=0.25, label='_nolegend_')
        ax.plot(dims, np.mean(ffc_dec, axis=0), color=colors[0])

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel('Classification Accuracy', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xlim([1, max(dims)])
        ax.set_xticks([1, int(max(dims)/2), int(max(dims))])
        ax.set_yticks([0., 0.15, 0.3])
        ax.legend(['FBC', 'FFC'], fontsize=10, loc='upper left', frameon=False)


        # Inset Plot 1
        pca_auc = np.sum(ffc_dec, axis=1)
        fca_auc = np.sum(fbc_dec, axis=1)
        _, p = scipy.stats.wilcoxon(pca_auc, fca_auc, alternative='less')

        axin = ax.inset_axes([0.6, 0.01, 0.35, 0.35])
        axin.scatter(np.zeros(len(pca_auc)), pca_auc, color='k', alpha=0.75, s=3)
        axin.scatter(np.ones(len(fca_auc)), fca_auc, color='r', alpha=0.75, s=3)
        axin.plot(np.array([(0, 1) for _ in range(ffc_dec.shape[0])]).T, np.array([(y1, y2) for y1, y2 in zip(np.sum(ffc_dec, axis=1), np.sum(fbc_dec, axis=1))]).T, color='k', alpha=0.5)
        axin.set_yticks([])
        axin.set_ylabel('Decoding AUC', fontsize=10)
        axin.set_xlim([-0.5, 1.5])
        axin.set_xticks([0, 1])
        axin.set_xticklabels(['FFC', 'FBC'], fontsize=10)

        fig.savefig('%s/Tsao_DecodingVDim_%s.pdf' % (figpath, region), bbox_inches='tight', pad_inches=0)


        ############# Main Plot 2
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        # Plot of the differences across dimensions
        ax.fill_between(dims, np.mean(fbc_dec - ffc_dec, axis=0) + np.std(fbc_dec - ffc_dec, axis=0)/np.sqrt(35),
                        np.mean(fbc_dec - ffc_dec, axis=0) - np.std(fbc_dec - ffc_dec, axis=0)/np.sqrt(35), color='blue', alpha=0.25)
        ax.plot(dims, np.mean(fbc_dec - ffc_dec, axis=0), color='blue')

        max_delta = np.max(np.mean(fbc_dec - ffc_dec, axis=0))
        fractional_delta = max_delta/np.mean(ffc_dec, axis=0)[np.argmax(np.mean(fbc_dec - ffc_dec, axis=0))]
        #print('Peak fractional improvement:%f' % fractional_delta)

        ax.set_xlabel('Dimension', fontsize=18)
        ax.set_ylabel(r'$\Delta$' + ' Classification Accuracy ', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.vlines(manual_dim, np.min(np.mean(fbc_dec - ffc_dec, axis=0)), np.mean(fbc_dec - ffc_dec, axis=0)[dims == manual_dim], linestyles='dashed', color='blue')
        ax.hlines(np.mean(fbc_dec - ffc_dec, axis=0)[dims == manual_dim], 0, manual_dim, linestyles='dashed', color='blue')
        ax.set_xlim([1, 30])
        ax.set_xticks([1, manual_dim, int(max(dims)/2), int(max(dims))])
        ax.set_yticks([0., 0.12])
        ax.set_ylim([-0.01, 0.125])
 
        fig.savefig('%s/Tsao_DecodingVDim_Delta_%s.pdf' % (figpath, region), bbox_inches='tight', pad_inches=0)
