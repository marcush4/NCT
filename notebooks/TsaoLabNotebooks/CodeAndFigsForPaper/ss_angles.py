import numpy as np
import scipy
import sys
import pdb
import matplotlib.pyplot as plt
from glob import glob
import pickle
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import KFold

from region_select import load_decoding_df, apply_df_filters
#sys.path.append('/home/akumar/nse/neural_control')
sys.path.append('/home/marcush/projects/neural_control/')
sys.path.append('/home/marcush/projects/neural_control/analysis_scripts/')
sys.path.append('/home/marcush/projects/github_repos')

if __name__ == '__main__':

    # # Where to save?
    if len(sys.argv) > 1:
        figpath = sys.argv[1]
    else:
        #figpath = '/home/akumar/nse/neural_control/figs/revisions'
        figpath = '/home/marcush/projects/neural_control/notebooks/TsaoLabNotebooks/CodeAndFigsForPaper/Figs'

    region = 'AM'
    dim = 21

    df, session_key = load_decoding_df(region)

    nFolds = len(np.unique(df['fold_idx']))
    sessions = np.unique(df[session_key].values)
    ss_angles = np.zeros((len(sessions), nFolds, dim))

    folds = np.arange(nFolds)
    dimvals = np.unique(df['dim'].values)

    for i, session in enumerate(sessions):
        for f, fold in enumerate(folds):

            df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':'PCA'}
            dfpca = apply_df_filters(df, **df_filter)
            df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':['LQGCA', 'FCCA']}
            dffcca = apply_df_filters(df, **df_filter)

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

    #fig_save_path = '/home/akumar/nse/neural_control/figs/revisions/ss_angles%s.pdf' % region
    fig_save_path = '%s/ss_angles%s.pdf' % (figpath, region)
    fig.savefig(fig_save_path)

    # Also generate here supplementary figures that communicate the full spread of angles
    dimvals = np.unique(df['dim'].values)[:-1]
    ssa1_median = np.zeros((len(sessions), len(dimvals), nFolds))
    ssa1_min = np.zeros((len(sessions), len(dimvals), nFolds))
    ssa1_max = np.zeros((len(sessions), len(dimvals), nFolds))

    ssa2_median = np.zeros((len(sessions), len(dimvals), nFolds))
    ssa2_min = np.zeros((len(sessions), len(dimvals), nFolds))
    ssa2_max = np.zeros((len(sessions), len(dimvals), nFolds))

    # Reference of what PCA looks like
    ssa3_mean = np.zeros((len(sessions), len(dimvals), nFolds))
    ssa3_min = np.zeros((len(sessions), len(dimvals), nFolds))
    ssa3_max = np.zeros((len(sessions), len(dimvals), nFolds))

    # SVD of concatenated projection
    joint_sv = np.zeros((len(sessions), len(dimvals), nFolds))

    for i, session in enumerate(sessions):
        for j, dim in enumerate(dimvals):
            for f in range(nFolds):
                df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':['LQGCA', 'FCCA']}
                dfd1_fc = apply_df_filters(df, **df_filter)
                df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':'PCA'}
                dfd1_pc = apply_df_filters(df, **df_filter)

                assert(dfd1_fc.shape[0] == 1)
                assert(dfd1_pc.shape[0] == 1)

                df_filter = {'dim':dim + 1, session_key:session, 'fold_idx':f, 'dimreduc_method':['LQGCA', 'FCCA']}
                #df_filter = {'dim':dimvals[j+1], session_key:session, 'fold_idx':f, 'dimreduc_method':['LQGCA', 'FCCA']}
                dfd2 = apply_df_filters(df, **df_filter)
                assert(dfd2.shape[0] == 1)

                ssa1 = scipy.linalg.subspace_angles(dfd1_fc.iloc[0]['coef'], dfd1_pc.iloc[0]['coef'][:, 0:dim])
                joint_proj = np.hstack([dfd1_fc.iloc[0]['coef'], dfd1_pc.iloc[0]['coef'][:, 0:dim]])
                s = np.linalg.svd(joint_proj, compute_uv=False)
                joint_sv[i, j, f] = np.sum(s) - dim
                ssa2 = scipy.linalg.subspace_angles(dfd1_fc.iloc[0]['coef'], dfd2.iloc[0]['coef'])

                df_filter = {'dim':dim +1, session_key:session, 'fold_idx':f, 'dimreduc_method':'PCA'}
                #df_filter = {'dim':dimvals[j+1], session_key:session, 'fold_idx':f, 'dimreduc_method':'PCA'}
                dfd2 = apply_df_filters(df, **df_filter)
                assert(dfd2.shape[0] == 1)
                ssa3 = scipy.linalg.subspace_angles(dfd1_pc.iloc[0]['coef'][:, 0:dim], dfd2.iloc[0]['coef'][:, 0:dim])

                r = {}
                r['session'] = session
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

    fig_save_path = '%s/jointsv_vdim%s.pdf' % (figpath, region)
    fig.savefig(fig_save_path, bbox_inches='tight', pad_inches=0)

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
         
    fig_save_path = '%s/ssa_vdim%s.pdf' % (figpath, region)
    fig.savefig(fig_save_path, bbox_inches='tight', pad_inches=0)
