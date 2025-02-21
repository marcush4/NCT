import numpy as np
import scipy
import sys
import pdb
import matplotlib.pyplot as plt
from glob import glob
import pickle
from pyuoi.linear_model.var import VAR
from tqdm import tqdm
import pandas as pd
from neurosim.models.var import VAR as VARss
from neurosim.models.var import form_companion
from copy import deepcopy
from sklearn.model_selection import KFold

from pseudopy import Normal

sys.path.append('/home/akumar/nse/neural_control')
from loaders import load_sabes, load_peanut, load_cv
from subspaces import estimate_autocorrelation
from utils import apply_df_filters
from dstableFGM import dstable_descent

if __name__ == '__main__':

    # # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        figpath = '/home/akumar/nse/neural_control/figs/revisions'

    M1 = False

    if M1:
        with open('/mnt/Secondary/data/postprocessed/sabes_M1subset2_dec_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
    else:
        with open('/mnt/Secondary/data/postprocessed/sabes_S1_subset2_dec_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)

    dim = 6
    data_files = np.unique(df['data_file'].values)
    ss_angles = np.zeros((len(data_files), 5, dim))

    folds = np.arange(5)
    dimreduc_methods = dimreduc_methods = ['PCA', 'LQGCA']
    LQGCA_dimreduc_args = [{'T':3, 'loss_type':'trace', 'n_init':10}]
    dimvals = np.unique(df['dim'].values)

    # Pick one
    decoder_arg = df.iloc[0]['decoder_args']
    df = apply_df_filters(df, decoder_args=decoder_arg)

    subset_index = 3
    if M1:
        subset_files = ['pca_subset_q75_M1.pkl', 'fca_subset_q75_M1.pkl', 
                    'fca_ldasubset_q75_M1.pkl', 'pca_ldasubset_q75_M1.pkl']
    else:
        subset_files = ['pca_subset_q75_S1.pkl', 'fca_subset_q75_S1.pkl', 
                    'fca_ldasubset_q75_S1.pkl', 'pca_ldasubset_q75_S1.pkl']

    subset_filter = [idx for idx in range(df.shape[0]) if df.iloc[idx]['loader_args']['subset_file'] == subset_files[subset_index]]
    df = df.iloc[subset_filter]

    for i, data_file in enumerate(data_files):
        for f, fold in enumerate(folds):

            dfpca = apply_df_filters(df, data_file=data_file, dimreduc_method='PCA', fold_idx=fold, dim=dim)
            dffcca = apply_df_filters(df, data_file=data_file, dimreduc_method='LQGCA', dimreduc_args=LQGCA_dimreduc_args[0], fold_idx=fold, dim=dim)

            try:
                assert(dfpca.shape[0] == 1)
            except:
                pdb.set_trace()

            assert(dffcca.shape[0] == 1)
            
            ss_angles[i, f, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:dim], dffcca.iloc[0]['coef'])

            # print('Calculating subspace angles between random projections...')
            # nrand = int(1e3)
            # rand = np.random.default_rng(42)
            # orth = scipy.stats.ortho_group
            # orth.random_state = rand
            # U = orth.rvs(dfpca.iloc[0]['coef'].shape[0], size=nrand)
            # U = U[..., 0:dim]
            # ss_angles = np.zeros((nrand, nrand, dim))
            # svd = np.zeros((nrand, nrand, 2 * dim))

            # Ujoint = []
            # print(U.shape)
            # for k1 in tqdm(range(nrand)):
            #     for k2 in range(nrand):
            #         ss_angles[k1, k2] = scipy.linalg.subspace_angles(U[k1], U[k2])
            #         joint_proj = np.hstack([U[k1], U[k2]])
            #         s = np.linalg.svd(joint_proj, compute_uv=False)
            #         svd[k1, k2] = s

            # pdb.set_trace()

    fig, ax = plt.subplots(figsize=(1, 4))

    medianprops = {'linewidth':0}
    bplot = ax.boxplot(np.mean(ss_angles, axis=-1).ravel(), patch_artist=True, medianprops=medianprops, notch=True, vert=True, showfliers=False)
    ax.set_xticks([])
    ax.set_ylabel('FCCA/PCA avg. ' + r'$\theta$ '+  '(rads)', fontsize=14)
    ax.set_yticks([0, np.pi/8, np.pi/4, 3 * np.pi/8, np.pi/2])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_yticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'], fontsize=14)

    for patch in bplot['boxes']:
        patch.set_facecolor('k')
        patch.set_alpha(0.75)

    if M1:
        fig.savefig('/home/akumar/nse/neural_control/figs/revisions/ss_anglesM1_subset%d.pdf' % subset_index)
    else:
        fig.savefig('/home/akumar/nse/neural_control/figs/revisions/ss_anglesS1_subset%d.pdf' % subset_index)

    # Also generate here supplementary figures that communicate the full spread of angles
    dimvals = np.unique(df['dim'].values)[:-1]
    ssa1_median = np.zeros((len(data_files), len(dimvals), 5))
    ssa1_min = np.zeros((len(data_files), len(dimvals), 5))
    ssa1_max = np.zeros((len(data_files), len(dimvals), 5))

    ssa2_median = np.zeros((len(data_files), len(dimvals), 5))
    ssa2_min = np.zeros((len(data_files), len(dimvals), 5))
    ssa2_max = np.zeros((len(data_files), len(dimvals), 5))

    # Reference of what PCA looks like
    ssa3_mean = np.zeros((len(data_files), len(dimvals), 5))
    ssa3_min = np.zeros((len(data_files), len(dimvals), 5))
    ssa3_max = np.zeros((len(data_files), len(dimvals), 5))

    # SVD of concatenated projection
    joint_sv = np.zeros((len(data_files), len(dimvals), 5))

    for i, data_file in enumerate(data_files):
        for j, dim in enumerate(dimvals):
            for f in range(5):
                dfd1_fc = apply_df_filters(df, dim=dim, data_file=data_file, fold_idx=f, dimreduc_method='LQGCA', dimreduc_args=LQGCA_dimreduc_args[0])
                dfd1_pc = apply_df_filters(df, dim=dim, data_file=data_file, fold_idx=f, dimreduc_method='PCA')

                assert(dfd1_fc.shape[0] == 1)
                assert(dfd1_pc.shape[0] == 1)

                dfd2 = apply_df_filters(df, dim=dim + 1, data_file=data_file, fold_idx=f, dimreduc_method='LQGCA', dimreduc_args=LQGCA_dimreduc_args[0])
                assert(dfd2.shape[0] == 1)

                ssa1 = scipy.linalg.subspace_angles(dfd1_fc.iloc[0]['coef'], dfd1_pc.iloc[0]['coef'][:, 0:dim])
                joint_proj = np.hstack([dfd1_fc.iloc[0]['coef'], dfd1_pc.iloc[0]['coef'][:, 0:dim]])
                s = np.linalg.svd(joint_proj, compute_uv=False)
                joint_sv[i, j, f] = np.sum(s) - dim
                ssa2 = scipy.linalg.subspace_angles(dfd1_fc.iloc[0]['coef'], dfd2.iloc[0]['coef'])

                dfd2 = apply_df_filters(df, dim=dim + 1, data_file=data_file, fold_idx=f, dimreduc_method='PCA')
                assert(dfd2.shape[0] == 1)
                ssa3 = scipy.linalg.subspace_angles(dfd1_pc.iloc[0]['coef'][:, 0:dim], dfd2.iloc[0]['coef'][:, 0:dim])

                r = {}
                r['data_file'] = data_file
                r['dim'] = dim
                r['fold'] = f
                r['ssa1'] = ssa1
                r['ssa2'] = ssa2

                ssa1_median[i, j, f] = np.median(ssa1)
                ssa1_min[i, j, f] = np.min(ssa1)
                ssa1_max[i, j, f] = np.max(ssa1)

                ssa2_median[i, j, f] = np.median(ssa2[0:dim])
                ssa2_min[i, j, f] = np.min(ssa2)
                ssa2_max[i, j, f] = np.max(ssa2)

                ssa3_mean[i, j, f] = np.mean(ssa3[0:dim])
                ssa3_min[i, j, f] = np.min(ssa3)
                ssa3_max[i, j, f] = np.max(ssa3)

    # New plot of just the joint singular values
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(dimvals, np.mean(joint_sv, axis=(0, -1)), color='b')
    ax.plot(dimvals, dimvals, color='k', linestyle='dashed')
    ax.set_ylabel('Effective Dimensionality of ' + r'$[V_{FBC}, V_{FFC}]$')
    ax.set_xlabel('Dimension')
    fig.tight_layout()

    if M1:
        fig.savefig('/home/akumar/nse/neural_control/figs/revisions/jointsv_vdim_subset%d.pdf' % subset_index, bbox_inches='tight', pad_inches=0)
    else:
        fig.savefig('/home/akumar/nse/neural_control/figs/revisions/jointsv_vdimS1_subset%d.pdf' % subset_index, bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(np.mean(np.mean(ssa1_median, axis=0), axis=-1), color='b', alpha=0.75, linestyle='-')
    ax[0].plot(np.mean(np.mean(ssa1_min, axis=0), axis=-1), color='b', alpha=0.75, linestyle='--')
    ax[0].plot(np.mean(np.mean(ssa1_max, axis=0), axis=-1), color='b', alpha=0.75, linestyle=':')

    ax[1].plot(np.mean(np.mean(ssa2_median, axis=0), axis=-1), color='b', alpha=0.75, linestyle='-')
    ax[1].plot(np.mean(np.mean(ssa2_min, axis=0), axis=-1), color='b', alpha=0.75, linestyle='--')
    ax[1].plot(np.mean(np.mean(ssa2_max, axis=0), axis=-1), color='b', alpha=0.75, linestyle=':')

    ax[0].set_ylim([0, np.pi/2])
    ax[0].set_yticks([0, np.pi/8, np.pi/4, 3 * np.pi/8, np.pi/2])
    ax[0].set_yticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])

    ax[0].set_ylabel('FCCA/PCA subspace angle')
    ax[0].set_xlabel('Dimension')
    ax[0].legend(['Median', 'Min', 'Max'])
        
    ax[1].set_ylabel('FCCA d/FCCA d + 1 subspace angle')
    ax[1].set_xlabel('Dimension')
    ax[1].legend(['Median', 'Min', 'Max'])
    ax[1].set_yticks([0, np.pi/8, np.pi/4, 3 * np.pi/8, np.pi/2])
    ax[1].set_yticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])

    # ax[1].plot(np.mean(np.mean(ssa3_mean, axis=0), axis=-1))
    # ax[1].plot(np.mean(np.mean(ssa3_min, axis=0), axis=-1))
    # ax[1].plot(np.mean(np.mean(ssa3_max, axis=0), axis=-1))
    fig.tight_layout()
    if M1:
        fig.savefig('/home/akumar/nse/neural_control/figs/revisions/ssa_vdim_subset%d.pdf' % subset_index, bbox_inches='tight', pad_inches=0)
    else:
        fig.savefig('/home/akumar/nse/neural_control/figs/revisions/ssa_vdimS1_subset%d.pdf' % subset_index, bbox_inches='tight', pad_inches=0)
