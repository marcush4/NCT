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

from config import PATH_DICT
from region_select import *

sys.path.append(PATH_DICT['repo'])
from utils import apply_df_filters

from FCCA.fcca import lqg_spectrum

dt = 1
# Amalgamate, calculate, and plot LQG spectrum results

# (1) SOC
with open(PATH_DICT['soc'] + '/soc_df.pkl', 'rb') as f:
    df = pickle.load(f)
with open(PATH_DICT['soc'] + '/Amats.pkl', 'rb') as f:
    Alist = pickle.load(f)

df = pd.DataFrame(df)

R = np.unique(df['R'].values)[0:20]
reps = np.unique(df['rep'].values)
inner_reps = np.unique(df['inner_rep'].values)
if not os.path.exists(PATH_DICT['tmp'] + '/lqg_spec_soc_tmp.pkl'):
    spect_fcca = np.zeros((R.size, reps.size, inner_reps.size, Alist[0][0].shape[0]))
    spect_pca = np.zeros(spect_fcca.shape)
    for i, r in tqdm(enumerate(R)):
        for j, rep in enumerate(reps):
            A = Alist[j][i]
            nn = np.linalg.norm(A @ A.T - A.T @ A)

            # Solve for the exact covarinace function and evaluate it at intervals of dt
            Pi = scipy.linalg.solve_continuous_lyapunov(A, -np.eye(A.shape[0]))
            t_ = [jj * dt for jj in range(10)]
            cross_covs = [scipy.linalg.expm(tau * A) @ Pi for tau in t_]

            cross_covs_rev = [np.linalg.inv(cross_covs[0]) @ c.T @ np.linalg.inv(cross_covs[0]) 
                              for c in cross_covs]

            cross_covs = torch.tensor(cross_covs)
            cross_covs_rev = torch.tensor(cross_covs_rev)

            for k, irep in enumerate(inner_reps):
                # Get the coefficient matrices
                df_ = apply_df_filters(df, R=r, rep=rep, inner_rep=irep)
                assert(df_.shape[0]) == 1
                
                Vpca = df_.iloc[0]['coef_pca']
                Vfcca = df_.iloc[0]['coef_fcca']
                # Assess LQG spectrum from FCCA
                spec1 = lqg_spectrum(torch.tensor(Vfcca), cross_covs, cross_covs_rev)
                spect_fcca[i, j, k] = spec1
                # Asssert LQG spectrum from PCA
                spec2 = lqg_spectrum(torch.tensor(Vpca), cross_covs, cross_covs_rev)
                spect_pca[i, j, k] = spec2
    with open(PATH_DICT['tmp'] + '/lqg_spec_soc_tmp.pkl', 'wb') as f:
        f.write(pickle.dumps(spect_fcca))
        f.write(pickle.dumps(spect_pca))

else:
    with open(PATH_DICT['tmp'] + '/lqg_spec_soc_tmp.pkl', 'rb') as f:
        spect_fcca = pickle.load(f)
        spect_pca = pickle.load(f)

# Average over inner reps and reps, plot for a particular R
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(spect_fcca[-1, 0, 0], color='r')
ax.plot(spect_pca[-1, 0, 0], color='k')
fig.savefig('lqg.png')

# (2) Poisson LDS
with open(PATH_DICT['soc'] + '/soc_poisson_df.pkl', 'rb') as f:
    df = pickle.load(f)
df = pd.DataFrame(df)

R = np.unique(df['R'].values)[0:20]
reps = np.unique(df['rep'].values)
inner_reps = np.unique(df['inner_rep'].values)
if not os.path.exists(PATH_DICT['tmp'] + '/lqg_spec_socp_tmp.pkl'):
    spect_fcca = np.zeros((R.size, reps.size, inner_reps.size, Alist[0][0].shape[0]))
    spect_pca = np.zeros(spect_fcca.shape)
    for i, r in tqdm(enumerate(R)):
        for j, rep in enumerate(reps):
            A = Alist[j][i]
            nn = np.linalg.norm(A @ A.T - A.T @ A)

            # Solve for the exact covarinace function and evaluate it at intervals of dt
            Pi = scipy.linalg.solve_continuous_lyapunov(A, -np.eye(A.shape[0]))
            t_ = [jj * dt for jj in range(10)]
            cross_covs = [scipy.linalg.expm(tau * A) @ Pi for tau in t_]

            cross_covs_rev = [np.linalg.inv(cross_covs[0]) @ c.T @ np.linalg.inv(cross_covs[0]) 
                              for c in cross_covs]

            cross_covs = torch.tensor(cross_covs)
            cross_covs_rev = torch.tensor(cross_covs_rev)

            for k, irep in enumerate(inner_reps):
                # Get the coefficient matrices
                df_ = apply_df_filters(df, R=r, rep=rep, inner_rep=irep)
                try:
                    assert(df_.shape[0]) > 0
                except:
                    spect_fcca[i, j, k] = np.nan
                    spect_pca[i, j, k] = np.nan
                Vpca = df_.iloc[0]['pca_coef']
                Vfcca = df_.iloc[0]['fcca_coef']
                # Assess LQG spectrum from FCCA
                spec1 = lqg_spectrum(torch.tensor(Vfcca), cross_covs, cross_covs_rev)
                spect_fcca[i, j, k] = spec1
                # Asssert LQG spectrum from PCA
                spec2 = lqg_spectrum(torch.tensor(Vpca), cross_covs, cross_covs_rev)
                spect_pca[i, j, k] = spec2
    with open(PATH_DICT['tmp'] + '/lqg_spec_socp_tmp.pkl', 'wb') as f:
        f.write(pickle.dumps(spect_fcca))
        f.write(pickle.dumps(spect_pca))

else:
    with open(PATH_DICT['tmp'] + '/lqg_spec_socp_tmp.pkl', 'rb') as f:
        spect_fcca = pickle.load(f)
        spect_pca = pickle.load(f)

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(spect_fcca[-1, 0, 0], color='r')
ax.plot(spect_pca[-1, 0, 0], color='k')
fig.savefig('lqg_p.png')

# (3) T.O RNN
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(spect_fcca[-1, 0, 0], color='r')
ax.plot(spect_pca[-1, 0, 0], color='k')
fig.savefig('lqg_p.png')
