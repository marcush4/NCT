import numpy as np
import sys, os
import pdb 
import pandas as pd
import pickle
import scipy
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold



import PSID
from PSID.evaluation import evalPrediction
from decoders import expand_state_space
from tqdm import tqdm

M1 = False
overwrite = False

if not os.path.exists('tmp'):
    os.makedirs('tmp')

if M1:
    tmp_exists = os.path.exists('tmp/psid_tmpM1.pkl')
else:
    tmp_exists = os.path.exists('tmp/psid_tmpS1.pkl')

if not tmp_exists or overwrite:
    if M1:
        with open('/mnt/Secondary/data/postprocessed/sabes_m1subtrunc_dec_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
    else:
        with open('/mnt/Secondary/data/postprocessed/sabes_s1subtrunc_dec_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)

    filt = [idx for idx in range(df.shape[0]) 
            if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
    df = df.iloc[filt]
    # Pick one
    decoder_arg = df.iloc[0]['decoder_args']
    df = apply_df_filters(df, decoder_args=decoder_arg)

    data_files = np.unique(df['data_file'].values)

    rl = []
    dimvals = np.arange(1, 31)
    for i, data_file in tqdm(enumerate(data_files)):
        if M1:
            dat = load_sabes('/mnt/Secondary/data/sabes/%s' % data_file)
        else:
            dat = load_sabes('/mnt/Secondary/data/sabes/%s' % data_file, region='S1')
        y = dat['spike_rates'].squeeze()
        z = dat['behavior']
        z, y = expand_state_space([z], [    y])
        z = z[0]
        y = y[0]

        for j, d in enumerate(dimvals):
            for fidx, (train_idxs, test_idxs) in enumerate(KFold(n_splits=5).split(y)):

                ytrain = y[train_idxs]
                ytest = y[test_idxs]

                ztrain = z[train_idxs]
                ztest = z[test_idxs]

                idsys = PSID.PSID(ytrain, ztrain, nx=d, n1=d, i=5)
                zpred, _, _ = idsys.predict(ytest)
                r2 = evalPrediction(ztest, zpred, 'R2')

                result = {}
                result['data_file'] = data_file
                result['dim'] = d
                result['fidx'] = fidx
                result['r2'] = r2

                rl.append(result)

    df1 = pd.DataFrame(rl)

    data_files = np.unique(df['data_file'].values)
    dimvals = np.arange(1, 31)
    rl2 = []
    for i, data_file in tqdm(enumerate(data_files)):

        if M1:
            dat = load_sabes('/mnt/Secondary/data/sabes/%s' % data_file)
        else:
            dat = load_sabes('/mnt/Secondary/data/sabes/%s' % data_file, region='S1')

        y = dat['spike_rates'].squeeze()
        z = dat['behavior']
        z, y = expand_state_space([z], [y])
        z = z[0]
        y = y[0]

        for j, d in enumerate(dimvals):
            for fidx, (train_idxs, test_idxs) in enumerate(KFold(n_splits=5).split(y)):

                ytrain = y[train_idxs]
                ytest = y[test_idxs]

                ztrain = z[train_idxs]
                ztest = z[test_idxs]

                df = apply_df_filters(df, data_file=data_file, dim=d, fold_idx=fidx, dimreduc_method='PCA')
                assert(df.shape[0] == 1)

                ytrain_pca = ytrain @ df.iloc[0]['coef'][:, 0:d]            
                ytest_pca = ytest @ df.iloc[0]['coef'][:, 0:d]

                idsys = PSID.PSID(ytrain_pca, ztrain, nx=6, n1=6, i=5)
                zpred, _, _ = idsys.predict(ytest_pca)
                r2_pca = evalPrediction(ztest, zpred, 'R2')

                df = apply_df_filters(df, data_file=data_file, dim=d, fold_idx=fidx, dimreduc_method='LQGCA')
                assert(df.shape[0] == 1)
                ytrain_fca = ytrain @ df.iloc[0]['coef'][:, 0:d]            
                ytest_fca = ytest @ df.iloc[0]['coef'][:, 0:d]

                idsys = PSID.PSID(ytrain_fca, ztrain, nx=6, n1=6, i=5)
                zpred, _, _ = idsys.predict(ytest_fca)
                r2_fca = evalPrediction(ztest, zpred, 'R2')

                result = {}
                result['data_file'] = data_file
                result['dim'] = d
                result['fidx'] = fidx
                result['r2_pca'] = r2_pca
                result['r2_fca'] = r2_fca

                rl2.append(result)

    df2 = pd.DataFrame(rl2)

    if M1:
        with open('tmp/psid_tmpM1.pkl', 'wb') as f:
            f.write(pickle.dumps(df1))
            f.write(pickle.dumps(df2))
    else:
        with open('tmp/psid_tmpS1.pkl', 'wb') as f:
            f.write(pickle.dumps(df1))
            f.write(pickle.dumps(df2))

else:
    if M1:
        with open('tmp/psid_tmpM1.pkl', 'rb') as f:
            df1 = pickle.load(f)
            df2 = pickle.load(f)
    else:
        with open('tmp/psid_tmpS1.pkl', 'rb') as f:
            df1 = pickle.load(f)
            df2 = pickle.load(f)
    

fig, ax = plt.subplots(figsize=(4, 4))
colors = ['black', 'red', '#781820', '#5563fa']
dim_vals = np.arange(1, 31)

# PSID r2
dims = np.arange(1, 31)
psid_r2_f = np.zeros((len(data_files), dims.size, 5))
for i, data_file in tqdm(enumerate(data_files)):
    for j, dim in enumerate(dims):               
        for f in range(5):
            dim_fold_df = apply_df_filters(df1, data_file=data_file, dim=dim, fidx=f)
            assert(dim_fold_df.shape[0] == 1)
            psid_r2_f[i, j, f] = np.mean(dim_fold_df.iloc[0]['r2'][2:4])

dims = np.arange(1, 31)
psid_r2_dr = np.zeros((len(data_files), dims.size, 5, 2))
for i, data_file in tqdm(enumerate(data_files)):
    for j, dim in enumerate(dims):               
        for f in range(5):
            dim_fold_df = apply_df_filters(df2, data_file=data_file, dim=dim, fidx=f)
            assert(dim_fold_df.shape[0] == 1)
            psid_r2_dr[i, j, f, 0] = np.mean(dim_fold_df.iloc[0]['r2_pca'][2:4])
            psid_r2_dr[i, j, f, 1] = np.mean(dim_fold_df.iloc[0]['r2_fca'][2:4])

# FCCA averaged over folds
fca_r2 = np.mean(psid_r2_dr[:, :, :, 1], axis=2)
# PCA
pca_r2 = np.mean(psid_r2_dr[:, :, :, 0], axis=2)

psid_r2 = np.mean(psid_r2_f, axis=2)

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

ax.fill_between(dim_vals, np.mean(psid_r2, axis=0) + np.std(psid_r2, axis=0)/np.sqrt(35),
                np.mean(psid_r2, axis=0) - np.std(psid_r2, axis=0)/np.sqrt(35), color=colors[2], alpha=0.25)
ax.plot(dim_vals, np.mean(psid_r2, axis=0), color=colors[2])

# Plot the paired differences
# ax.plot(dim_vals, np.mean(fca_r2 - pca_r2, axis=0))
# ax.fill_between(dim_vals, np.mean(fca_r2 - pca_r2, axis=0) + np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35),
#                 np.mean(fca_r2 - pca_r2, axis=0) - np.std(fca_r2 - pca_r2, axis=0)/np.sqrt(35), color=colors[0], alpha=0.25)

ax.set_xlabel('Dimension', fontsize=18)
ax.set_ylabel('Velocity Decoding ' + r'$r^2$', fontsize=18)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

ax.legend(['FCCA', 'PCA', 'PSID'], fontsize=10, frameon=False, loc='lower right')
if M1:
    fig.savefig('/home/akumar/nse/neural_control/figs/revisions/psid_M1.pdf', bbox_inches='tight', pad_inches=0)
else:
    fig.savefig('/home/akumar/nse/neural_control/figs/revisions/psid_S1.pdf', bbox_inches='tight', pad_inches=0)
