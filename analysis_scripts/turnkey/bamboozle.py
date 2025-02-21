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

import matplotlib.cm as cm
import matplotlib.colors as colors

from region_select import *
from decodingvdim import get_decoding_performance
from config import PATH_DICT
sys.path.append(PATH_DICT['repo'])
from utils import calc_loadings, calc_cascaded_loadings

# Make a plot of 
# (1) avg. FCCA/PCA subspace angles vs. peak/integraated delta decoding
# (2) avg. FCCA/PCA subspace angles vs. avg. classification accuracy across quantiles
dim_dict = {
    'M1': 6,
    'S1': 3,
    'M1_trialized':6,
    'HPC_peanut': 11,
    'M1_maze':6,
    'AM': 21,
    'ML': 21
}

regions = ['M1', 'S1', 'M1_maze', 'HPC_peanut', 'M1_trialized']
region_labels = ['M1', 'S1', 'M1 maze', 'HPC', 'M1 trialized']
if not os.path.exists(PATH_DICT['tmp'] + '/bamboozle_tmp.pkl'):
    ss_angles_across_regions = []
    dr2_across_sessions = []
    cls_scores_across_sessions = []
    for region in regions:
        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        dim = dim_dict[region]
        sessions = np.unique(df[session_key].values)

        # getting subspace angles
        ss_angles = np.zeros((len(sessions), 5, dim))
        folds = np.arange(5)
        dimvals = np.unique(df['dim'].values)
        for i, session in enumerate(sessions):
            for f, fold in enumerate(folds):

                df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':'PCA'}
                dfpca = apply_df_filters(df, **df_filter)
                df_filter = {'dim':dim, session_key:session, 'fold_idx':f, 'dimreduc_method':['LQGCA', 'FCCA']}
                dffcca = apply_df_filters(df, **df_filter)
                assert(dfpca.shape[0] == 1)
                assert(dffcca.shape[0] == 1)
                
                ss_angles[i, f, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:dim], dffcca.iloc[0]['coef'])
        ss_angles_across_regions.append(np.mean(ss_angles, axis=-1))

        # getting delta_decoding

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
        
        dr2 = r2fc - r2pca
        dr2_across_sessions.append(dr2)

        # getting average classification accuracys
        savepath = '/psth_clustering_tmp%s.pkl' % region
        with open(PATH_DICT['tmp'] + savepath, 'rb') as f:
            scores = pickle.load(f)
            dummy_scores = pickle.load(f)
            random_scores = pickle.load(f)
            loadings_df = pickle.load(f)
            xall = pickle.load(f)
            u = pickle.load(f)

        sessions = np.unique(loadings_df[session_key])
        # Average across fbc fraction and session
        cls_scores_across_sessions.append(np.mean(scores, axis=(0, 2)))

    # Save as tmp to avoid re-acquiring data
    with open(PATH_DICT['tmp'] + '/bamboozle_tmp.pkl', 'wb') as f:
        f.write(pickle.dumps(ss_angles_across_regions))
        f.write(pickle.dumps(dr2_across_sessions))
        f.write(pickle.dumps(cls_scores_across_sessions))
else:
    with open(PATH_DICT['tmp'] + '/bamboozle_tmp.pkl', 'rb') as f:
        ss_angles_across_regions = pickle.load(f)
        dr2_across_sessions = pickle.load(f)
        cls_scores_across_sessions = pickle.load(f)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# (1) average subspace angles vs decoding delta (errorbar)
# x = []
# xerr = []
# y1 = []
# y1err = []
# y2 = []
# y2err = []
markers = ['o', 's', 'o', 'o', 'o', 's']
colors = ['purple', 'purple', 'b', 'g', '#f59b42', '#f59b42']


xall = []
y1all = []
y2all = []

for i, region in enumerate(regions):
    ss_angles = ss_angles_across_regions[i]
    # x.append(np.mean(ss_angles))
    # xerr.append(np.std(ss_angles))

    x = np.mean(ss_angles)
    xerr = np.std(ss_angles)
    dr2 = dr2_across_sessions[i][:, 0:30, :]
    dr2 = np.mean(dr2, axis=2)
    # Sum across dimensions
    dr2_auc = np.sum(dr2, axis=1)
    # y1.append(np.mean(dr2_auc))
    # y1err.append(np.std(dr2_auc))
    y1 = np.mean(dr2_auc)
    y1err = np.std(dr2_auc)

    # y2.append(np.mean(cls_scores_across_sessions[i]))
    # y2err.append(np.std(cls_scores_across_sessions[i]))
    y2 = np.mean(cls_scores_across_sessions[i])
    y2err = np.std(cls_scores_across_sessions[i])

    ax[0].errorbar(x, y1, xerr=xerr, yerr=y1err, color=colors[i],  
                   fmt='', marker=markers[i], label=region_labels[i], alpha=0.5)
    ax[1].errorbar(x, y2, xerr=xerr, yerr=y2err, color=colors[i],  
                   fmt='', marker=markers[i], label=region_labels[i], alpha=0.5)

    xall.append(np.mean(ss_angles, axis=1))
    y1all.append(dr2_auc)
    y2all.append(cls_scores_across_sessions[i])
 
x = np.concatenate(xall) * 180/np.pi
y1 = np.concatenate(y1all)
y2 = np.concatenate(y2all)

# Pearson correlation coefficients
r1, p1 = scipy.stats.pearsonr(x, y1)
r2, p2 = scipy.stats.pearsonr(x, y2)

print(f'dr2 auc linear correlation: {r1}, p={p1}')
print(f'classification accuracy linear correlation: {r2}, p={p2}')

r1, p1 = scipy.stats.spearmanr(x, y1)
r2, p2 = scipy.stats.spearmanr(x, y2)

print(f'dr2 auc spearman correlation: {r1}, p={p1}')
print(f'classification accuracy spearman correlation: {r2}, p={p2}')


ax[0].legend()
ax[1].legend()
ax[0].set_xlabel('FFC/FBC subspace angles (rads)')
ax[1].set_xlabel('FFC/FBC subspace angles (rads)')
ax[0].set_ylabel(r'$\Delta$' + ' Behavioral Decoding AUC')
ax[1].set_ylabel('Avg. FBC neuron classification acc.')
fig.tight_layout()
fig.savefig(PATH_DICT['figs'] + '/ssa_diagonstic.pdf')

# Also produce the raw scatter to inspect the correlation values
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(x, y1)
ax[1].scatter(x, y2)
fig.savefig(PATH_DICT['figs'] + '/ssa_diagonstic_raw_scatter.pdf')
