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

import umap
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from dca.cov_util import calc_cross_cov_mats_from_data
import matplotlib.cm as cm
import matplotlib.colors as colors

from region_select import *
from config import PATH_DICT
sys.path.append(PATH_DICT['repo'])
from utils import calc_loadings, calc_cascaded_loadings

from Fig4 import get_loadings_df

hist_ylims = {
    'M1': [0, 500],
    'S1': [0, 105],
    'HPC_peanut':[0, 50],
    'VISp':[0, 125],
    'AM':[0, 125],
    'ML':[0, 125]
}

hist_yticks = {
    'M1': [0, 150, 300],
    'S1': [0, 50, 100],
    'HPC_peanut':[0, 25, 50],
    'VISp': [0, 75, 125],
    'AM': [0, 75, 125],
    'ML': [0, 75, 125]
}

scatter_xlims = {
    'M1': [-0.05, 3.5],
    'S1': [-0.05, 3.5],
    'HPC_peanut':[-0.05, 3.5],
    'VISp': [-0.05, 2],
    'AM': [-0.05, 3.2],
    'ML': [-0.05, 3]
}

scatter_xticks = {
    'M1': [0, 3.5],
    'S1': [0, 3.5],
    'HPC_peanut': [0, 3.5],
    'VISp': [0, 2],
    'AM': [0, 3],
    'ML': [0, 3]
}

scatter_ylims = {
    'M1': [-0.05, 3.0],
    'S1': [-0.05, 2.0],
    'HPC_peanut':[-0.05, 2.0],
    'VISp': [-0.05, 1.2],
    'AM': [0.05, 1.2],
    'ML': [0.05, 1.5]
}

scatter_yticks = {
    'M1': [0., 3.0],
    'S1': [0., 2.0],
    'HPC_peanut': [0., 2.0],
    'VISp': [0., 1.0],
    'AM': [0., 1.0],
    'ML': [0., 1.0]
}

def calc_psth_su_stats(xall):

    def ccm_thresh(ccm):
        ccm = ccm.squeeze()
        # Normalize
        ccm /= ccm[0]
        thr = 1e-1
        acov_crossing = np.where(ccm < thr)
        if len(acov_crossing[0]) > 0:
            act = np.where(ccm < thr)[0][0]
        else:
            act = len(ccm)

        return act

    n_neurons = len(xall)
    nt = xall[0].shape[1]
    # Stats - dynamic range, autocorrelation time, FFT
    # Consider trial averaged and non-trial averaged variants
    dyn_range = np.zeros((n_neurons, 2))
    act = np.zeros((n_neurons, 2))
    fft = np.zeros((n_neurons, 2))    

    ccmT = int(min(20, xall[0].shape[-1]//2))
    for i in tqdm(range(len(xall))):
        dyn_range[i, 0] = np.max(np.abs(np.mean(xall[i], axis=0)))
        dyn_range[i, 1] = np.mean(np.max(np.abs(xall[i]), axis=1))

        ccm1 = calc_cross_cov_mats_from_data(np.mean(xall[i], axis=0)[:, np.newaxis], ccmT)
        ccm2 = calc_cross_cov_mats_from_data(xall[i][..., np.newaxis], ccmT)
        act[i, 0] = ccm_thresh(ccm1)
        act[i, 1] = ccm_thresh(ccm2)

        # FFT what's the feature? - total power contained beyond the DC component
        N = xall[i].shape[1]
        xfft = scipy.fft.fft(xall[i], axis=1)
        xpsd = np.mean(np.abs(xfft)**2, axis=0)[0:N//2]
        xpsd /= xpsd[0]

        fft[i, 0] = np.sum(xpsd[1:])
        # Trial average and then FFT
        xfft = scipy.fft.fft(np.mean(xall[i], axis=0))
        xpsd = np.abs(xfft**2)[0:N//2]
        xpsd /= xpsd[0]
        fft[i, 1] = np.sum(xpsd[1:])

    return dyn_range, act, fft

def plot_feature_scatter(df, data_path, session_key, region, dim, figpath):

    # Get corresponding loadings
    loadings_df = get_loadings_df(df, session_key, dim=dim)
    sessions = np.unique(loadings_df[session_key].values)
    # Relative FBC/FFC score
    rfbc = np.divide(loadings_df['FCCA_loadings'].values,
                        loadings_df['FCCA_loadings'].values +\
                        loadings_df['PCA_loadings'].values)    
    rffc = np.divide(loadings_df['PCA_loadings'].values,
                        loadings_df['FCCA_loadings'].values +\
                        loadings_df['PCA_loadings'].values)    

    # Load trialized spike rates
    xall = []
    print('Collecting PSTH')
    # Non-trial averaged psth
    for h, session in enumerate(sessions):
        # For ML/AL, need this full_arg_tuple
        if 'full_arg_tuple' in df.keys() and region in ['AM', 'ML']:
            # Modify the full_arg_tuple according to the desired loader args
            df_sess = apply_df_filters(df, **{session_key:session, 'loader_args':{'region': region}})
            full_arg_tuple = dict(df_sess.iloc[0]['full_arg_tuple'])
            # Do not boxcox
            full_arg_tuple['boxcox'] = None
            full_arg_tuple = [tuple(full_arg_tuple.items())]
        else:
            full_arg_tuple = None

        # Do not boxcox
        
        load_idx = loader_kwargs[region]['load_idx']  
        unique_loader_args = list({make_hashable(d) for d in df['loader_args']})
        loader_args = dict(unique_loader_args[load_idx])
        loader_args['boxcox'] = None
        x = get_rates_smoothed(data_path, region, session,
                        loader_args=loader_args, 
                        trial_average=False, full_arg_tuple=full_arg_tuple)
        xall.extend(x)

    # Sort nidx by highest rfbc and rffc
    # Then calculate the FFT associated with these neurons to 
    # get a sense of what features to use    
    # rffc_top = np.argsort(rffc)[::-1][0:10]
    # rfbc_top = np.argsort(rfbc)[::-1][0:10]

    # fig, ax = plt.subplots(10, 2, figsize=(8, 40))
    # for i, idx in enumerate(rffc_top):
    #     # Average after performing on trials
    #     N = xall[idx].shape[1]
    #     xfft = scipy.fft.fft(xall[idx], axis=1)
    #     freq = scipy.fft.fftfreq(N)[:N//2]
    #     a = ax[i, 0]
    #     a.plot(freq, np.mean(np.abs(xfft)**2, axis=0)[0:N//2])

    #     # Trial average and then FFT
    #     xfft = scipy.fft.fft(np.mean(xall[idx], axis=0))
    #     a = ax[i, 1]
    #     a.plot(freq, np.abs(xfft[0:N//2])**2)

    # fig.savefig('psth_fft_plots/top_ffc_fft.pdf', bbox_inches='tight', pad_inches=0)
    # plt.close(fig)
    # fig, ax = plt.subplots(10, 2, figsize=(8, 40))
    # for i, idx in enumerate(rfbc_top):
    #     # Average after performing on trials
    #     N = xall[idx].shape[1]
    #     xfft = scipy.fft.fft(xall[idx], axis=1)
    #     freq = scipy.fft.fftfreq(N)[:N//2]
    #     a = ax[i, 0]
    #     a.plot(freq, np.mean(np.abs(xfft)**2, axis=0)[0:N//2])

    #     # Trial average and then FFT
    #     xfft = scipy.fft.fft(np.mean(xall[idx], axis=0))
    #     a = ax[i, 1]
    #     a.plot(freq, np.abs(xfft[0:N//2])**2)

    # fig.savefig('psth_fft_plots/top_fbc_fft.pdf', bbox_inches='tight', pad_inches=0)
    # plt.close(fig)

    # # Calculate statistics on trialized firing rates
    dyn_range, act, fft = calc_psth_su_stats(xall)
    # Scatter in 3D - color by rfbc        
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    cmap_new = truncate_colormap(cm.RdGy_r, 0., 0.9)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(dyn_range[:, 0], act[:, 0], fft[:, 0],
                c=rfbc, edgecolors=(0.6, 0.6, 0.6, 0.6), 
                linewidth=0.01, s=15, cmap=cmap_new)
    ax.set_xlabel('Dynamic Range')
    ax.set_ylabel('Autocorrelation Time')
    ax.set_zlabel('PSD non-DC')
    fig.savefig('psth_scatter_0_%s.pdf' % region)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot()
    ax.scatter(dyn_range[:, 0], fft[:, 0],
                c=rfbc, edgecolors=(0.6, 0.6, 0.6, 0.6), 
                linewidth=0.01, s=15, cmap=cmap_new)

    ax.set_ylim(scatter_ylims[region])
    ax.set_xlim(scatter_xlims[region])

    ax.set_xticks(scatter_xticks[region])
    ax.set_yticks(scatter_yticks[region])
    
    ax.set_xlabel('Peak Amplitude', fontsize=12)
    ax.set_ylabel('Oscillation Strength', fontsize=12)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Rel. FBC Importance', fontsize=12)


    # ax.set_xlabel('Dynamic Range')
    # # ax.set_ylabel('Autocorrelation Time')
    # ax.set_ylabel('PSD non-DC')

    fig.savefig('%s/psth_scatter2D_0_%s.pdf' % (figpath, region),
                 bbox_inches='tight', pad_inches=0)

    return loadings_df, dyn_range, act, fft

def plot_lda(df, loadings_df, dyn_range, act, fft, session_key):

    # can we use these features with LDA?
    features = np.hstack([dyn_range[:, 0:1], act[:, 0:1], fft[:, 0:1]])
    scores, dummy_scores, random_scores, xtrans, ntype = lda_fit(df, loadings_df, session_key, features)
    sessions = np.unique(loadings_df[session_key])
    fbc_fraction = np.linspace(0.5, 0.95, 25)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    ax.plot(fbc_fraction, np.nanmean(scores, axis=(1, 2)), label="Data")
    ax.fill_between(fbc_fraction, np.nanmean(scores, axis=(1, 2)) - np.nanstd(scores, axis=(1, 2)), 
                    np.nanmean(scores, axis=(1, 2)) + np.nanstd(scores, axis=(1, 2)), alpha=0.5, label="_nolegend_")

    # Omitted for discussion with editor
    ax.plot(fbc_fraction, np.nanmean(dummy_scores, axis=(1, 2), label="Dummy"))
    ax.fill_between(fbc_fraction, np.nanmean(dummy_scores, axis=(1, 2)) - np.nanstd(dummy_scores, axis=(1, 2)), 
                    np.nanmean(dummy_scores, axis=(1, 2)) + np.nanstd(dummy_scores, axis=(1, 2)), alpha=0.5, label="_nolegend_")

    ax.plot(fbc_fraction, np.nanmean(random_scores, axis=(1, 2, 3)), color='purple', label="Random")
    ax.fill_between(fbc_fraction, np.nanmean(random_scores, axis=(1, 2, 3)) - np.nanstd(random_scores, axis=(1, 2, 3)), 
                    np.nanmean(random_scores, axis=(1, 2, 3)) + np.nanstd(random_scores, axis=(1, 2, 3)), alpha=0.5, color='purple', label="_nolegend_")

    ax.legend(['Data', 'Dummy', 'Random'], loc='lower right')
    ax.set_xlabel('FBC Quantile', fontsize=14)
    ax.set_ylabel('Classification Accuracy', fontsize=14)
    ax.set_ylim([0.5, 1])
    figpath = PATH_DICT['figs'] + '/psth_feature_classification_%s.pdf' % region
    fig.savefig(figpath, bbox_inches='tight', pad_inches=0)


    carray = cm.RdGy(range(256))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    N, bins, patches = ax.hist(xtrans, linewidth=1, bins=50)
    xbinned = np.digitize(xtrans, bins).squeeze()

    # Set each rectangle to a color gradient set by the fraction of entries that belong to each type
    fracs = []
    for i in range(len(patches)):
        x_ = np.where(xbinned == i + 1)[0]
        if len(x_) > 0:
            ntype_ = np.array(ntype)[x_]
            frac = np.sum(ntype_)/x_.size
            patches[i].set_facecolor(carray[int(255 * frac)])
            patches[i].set_edgecolor(carray[int(255*frac)])
            fracs.append(frac)
        else:
            fracs.append(np.nan)
        #patches[i].set_facecolor(carray[0])
    ax.set_xlabel('LDA Dimension 1')
    ax.set_ylabel('Count')
    ax.set_ylim(hist_ylims[region])
    ax.set_yticks(hist_yticks[region])
    #ax.set_aspect('equal')
    # Vertical colorbar

    cmap = cm.RdGy.reversed()
    norm = colors.Normalize(vmin=0, vmax=1)
    cbar= fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, orientation='vertical', label='', ticks=[0, 0.5, 1.])
    cbar.ax.set_yticklabels([])
    cbar.ax.set_ylabel('Rel. FBC Importance')
    #cbar.ax.set_yticks([0, 0.5, 1.0])

    # Add an inset - disabled for resubmission discussion with editor
    # ax.set_title('%s \n %s \n %s' % (la, deca, dra))
    # fig.tight_layout()

    # axin = ax.inset_axes([0.35, 0.6, 0.4, 0.3])
    # # axin.set_aspect(1)
    # axin.set_ylim([0.5, 1.0])
    # axin.set_xlim([0.5, 0.9])
    # axin.set_yticks([0.5, 1.0])
    # axin.set_xticks([0.5, 0.7, 0.9])
    # axin.plot(fbc_fraction, np.mean(scores, axis=(1, 2)), color='#625487')
    # axin.fill_between(fbc_fraction, np.mean(scores, axis=(1, 2)) - np.std(scores, axis=(1, 2)), np.mean(scores, axis=(1, 2)) + np.std(scores, axis=(1, 2)), alpha=0.5, color='#625487')

    figpath = PATH_DICT['figs'] + '/%s_psth_lda_viz.pdf' % region
    fig.savefig(figpath, bbox_inches='tight', pad_inches=0)

def lda_fit(dimreduc_df, loadings_df, session_key, u):
    fbc_fraction = np.linspace(0.5, 0.95, 25)
    ncv = 3
    nrandom = 100
    # Partition by data file
    sessions = np.unique(loadings_df[session_key].values)
    scores = np.zeros((len(fbc_fraction), len(sessions), ncv))
    dummy_scores = np.zeros((len(fbc_fraction), len(sessions), ncv))
    random_scores = np.zeros((len(fbc_fraction), len(sessions), ncv, nrandom))
    keys = ['FCCA_loadings', 'PCA_loadings']
    # Get indices from loadings_df
    indices = list(np.cumsum([loadings_df.loc[loadings_df[session_key] == s].shape[0]
                              for s in sessions]))
    indices.insert(0, 0)

    for ii, fbcf in tqdm(enumerate(fbc_fraction)):
        for i in range(len(sessions)):

            # Per recording session
            yf_ = []
            yp_ = []

            # Is a neuron more FBC or FFC?
            ntype = []

            for j in range(2):
                df = apply_df_filters(loadings_df, **{session_key:sessions[i]})
                x1 = df[keys[j]].values
                xx = []

                if j == 0:
                    yf_.append(x1)
                else:
                    yp_.append(x1)

            rfbc = yf_[-1]/(yf_[-1] + yp_[-1])
            rffc = yp_[-1]/(yf_[-1] + yp_[-1])

            # do this by quantile
            cutoff = np.quantile(rfbc, fbcf)
            for n in range(rfbc.size):
                if rfbc[n] > cutoff:
                    ntype.append(0)
                else:
                    ntype.append(1)

            # get the neurons assocaited with the session
            u_i = u[indices[i]:indices[i+1], :]
            #logreg = LogisticRegression()
            nanmask = np.ma.masked_where(np.isnan(u_i), u_i).mask
            if np.any(nanmask):
                u_i = u_i[~np.any(nanmask, axis=1)]
                ntype = np.array(ntype)[~np.any(nanmask, axis=1)]
            lda = LinearDiscriminantAnalysis(n_components=1)
            # perform cross-validation
            # for higher values of fbc fraction, may not have enough data so ignore
            try:
                scores[ii, i] = cross_val_score(lda, u_i, ntype, cv=ncv)
                dummy_scores[ii, i] = cross_val_score(DummyClassifier(strategy='stratified'), u_i, ntype, cv=ncv)
                # Compare also to 100 random assignments of neuron types
                for k in range(nrandom):
                    ntype_rand = np.random.permutation(ntype)
                    random_scores[ii, i, :, k] = cross_val_score(lda, u_i, ntype_rand, cv=ncv)
            except:
                scores[ii, i] = np.nan
                dummy_scores[ii, i] = np.nan
                # Compare also to 100 random assignments of neuron types
                random_scores[ii, i] = np.nan

    # Also fit all at once for visualization purposes

    keys = ['FCCA_loadings', 'PCA_loadings']
    fbcf = 0.5
    # Is a neuron more FBC or FFC?
    ntype = []
    for i in range(len(sessions)):

        # Per recording session
        yf_ = []
        yp_ = []

        for j in range(2):
            df = apply_df_filters(loadings_df, **{session_key:sessions[i]})
            x1 = df[keys[j]].values
            xx = []

            if j == 0:
                yf_.append(x1)
            else:
                yp_.append(x1)

        rfbc = yf_[-1]/(yf_[-1] + yp_[-1])
        rffc = yp_[-1]/(yf_[-1] + yp_[-1])

        # do this by quantile
        cutoff = np.quantile(rfbc, fbcf)
        for n in range(rfbc.size):
            if rfbc[n] > cutoff:
                ntype.append(0)
            else:
                ntype.append(1)
    lda = LinearDiscriminantAnalysis(n_components=1)

    # Deal with nans
    nanmask = np.ma.masked_where(np.isnan(u), u).mask
    
    if np.any(nanmask):
        u = u[~np.any(nanmask, axis=1)]
        ntype = np.array(ntype)[~np.any(nanmask, axis=1)]    
    xtrans = lda.fit_transform(u, ntype)
    return scores, dummy_scores, random_scores, xtrans, ntype

def run_umap_3d(dimreduc_df, data_path, session_key, region, dim, boxcox=False,
                     overwrite=False):
    if boxcox:
        savepath = '/umap3d_clustering_tmp%s_boxcox.pkl' % region
    else:
        savepath = '/umap3d_clustering_tmp%s.pkl' % region

    if not os.path.exists(PATH_DICT['tmp'] + savepath) or overwrite:
        xall = []
        print('Collecting PSTH')

        if region in ['AM', 'ML']:
            dimreduc_reg_df = apply_df_filters(dimreduc_df, **{'loader_args':{'region': region}})
            loadings_df = get_loadings_df(dimreduc_reg_df, session_key, dim=dim)
        else:
            loadings_df = get_loadings_df(dimreduc_df, session_key, dim=dim)

        sessions = np.unique(loadings_df[session_key].values)
        for h, session in enumerate(sessions):
            if region in ['AM', 'ML']:  
                x = get_rates_smoothed(data_path, region, session, full_arg_tuple=dimreduc_df_sess['full_arg_tuple'])
            else:
                x = get_rates_smoothed(data_path, region, session,
                              loader_args=dimreduc_df.iloc[0]['loader_args'])
            xall.append(x)

        # MIH HACK TO CHANGE LATER !!!
        if region in ['AM', 'ML']:
            xall = [x[:, :19] for x in xall]

        xall_stacked = np.vstack(xall)
        param = (3, 0.2, 50)
        # Share the UMAP projection across recording sessions - measure classificationa ccuraacy per recording session
        if region == 'AM':
            fit = umap.UMAP(min_dist=param[1], n_neighbors=param[2], n_components=param[0], random_state=None, transform_seed=None)
        else:
            fit = umap.UMAP(min_dist=param[1], n_neighbors=param[2], n_components=param[0], random_state=42, transform_seed=42)
        print('Fiting UMAP')
        u = fit.fit_transform(xall_stacked)        

        scores, dummy_scores, random_scores = lda_fit(dimreduc_df, loadings_df, session_key, u)

        with open(PATH_DICT['tmp'] + savepath, 'wb') as f:
            f.write(pickle.dumps(scores))
            f.write(pickle.dumps(dummy_scores))
            f.write(pickle.dumps(random_scores))
            f.write(pickle.dumps(loadings_df))
            f.write(pickle.dumps(xall))
            f.write(pickle.dumps(u))
            # f.write(pickle.dumps(class_sizes))

# Replicate the LDA plots but with UMAP --> 3D, and then also 
# provide a raw scatter
def plot_umap_3d(dimreduc_df, session_key, region='M1', quantile=0.75, boxcox=False):
    if boxcox:
        savepath = '/umap3d_clustering_tmp%s_boxcox.pkl' % region
    else:
        savepath = '/umap3d_clustering_tmp%s.pkl' % region

    if not os.path.exists(PATH_DICT['tmp'] + savepath):
        raise ValueError('Call run_umap_3d first')

    with open(PATH_DICT['tmp'] + savepath, 'rb') as f:
        scores = pickle.load(f)
        dummy_scores = pickle.load(f)
        random_scores = pickle.load(f)
        loadings_df = pickle.load(f)
        xall = pickle.load(f)
        u = pickle.load(f)

    sessions = np.unique(loadings_df[session_key])
    fbc_fraction = np.linspace(0.5, 0.95, 25)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(fbc_fraction, np.mean(scores, axis=(1, 2)))
    ax.fill_between(fbc_fraction, np.mean(scores, axis=(1, 2)) - np.std(scores, axis=(1, 2)), np.mean(scores, axis=(1, 2)) + np.std(scores, axis=(1, 2)), alpha=0.5)

    # Omitted for discussion with editor
    ax.plot(fbc_fraction, np.mean(dummy_scores, axis=(1, 2)))
    ax.fill_between(fbc_fraction, np.mean(dummy_scores, axis=(1, 2)) - np.std(dummy_scores, axis=(1, 2)), np.mean(dummy_scores, axis=(1, 2)) + np.std(dummy_scores, axis=(1, 2)), alpha=0.5)

    ax.plot(fbc_fraction, np.mean(random_scores, axis=(1, 2, 3)), color='purple')
    ax.fill_between(fbc_fraction, np.mean(random_scores, axis=(1, 2, 3)) - np.std(random_scores, axis=(1, 2, 3)), 
                    np.mean(random_scores, axis=(1, 2, 3)) + np.std(random_scores, axis=(1, 2, 3)), alpha=0.5, color='purple')

    ax.legend(['Data', 'Dummy', 'Random'], loc='lower right')
    ax.set_xlabel('FBC Quantile', fontsize=14)
    ax.set_ylabel('Classification Accuracy', fontsize=14)
    ax.set_ylim([0.5, 1])


    # ax[1].plot(fbc_fraction, np.mean(class_sizes[:, :, 0], axis=1), color='r')
    # ax[1].plot(fbc_fraction, np.mean(class_sizes[:, :, 1], axis=1), color='k')
    # ax[1].set_xlabel('FBC Quantile', fontsize=14)
    # ax[1].set_ylabel('Average Class Size', fontsize=14)
    # fig.tight_layout()
    if boxcox:
        figpath = PATH_DICT['figs'] + '/umap_3dclusteringLDA%s_boxcox.pdf' % region
    else:
        figpath = PATH_DICT['figs'] + '/umap_3dclusteringLDA%s.pdf' % region

    fig.savefig(figpath, bbox_inches='tight', pad_inches=0)

    # Run LDA on everything aggregated and visualize
    # Visualize results using LDA

    ncv = 5
    nrandom = 100
    keys = ['FCCA_loadings', 'PCA_loadings']
    indices = list(np.cumsum([len(x_) for x_ in xall]))
    indices.insert(0, 0)
    fbcf = 0.5
    # Is a neuron more FBC or FFC?
    ntype = []
    for i in range(len(xall)):

        # Per recording session
        yf_ = []
        yp_ = []


        for j in range(2):
            df = apply_df_filters(loadings_df, **{session_key:sessions[i]})
            x1 = df[keys[j]].values
            xx = []

            if j == 0:
                yf_.append(x1)
            else:
                yp_.append(x1)

        rfbc = yf_[-1]/(yf_[-1] + yp_[-1])
        rffc = yp_[-1]/(yf_[-1] + yp_[-1])

        # do this by quantile
        cutoff = np.quantile(rfbc, fbcf)
        for n in range(rfbc.size):
            if rfbc[n] > cutoff:
                ntype.append(0)
            else:
                ntype.append(1)
    
    lda = LinearDiscriminantAnalysis(n_components=1)
    xtrans = lda.fit_transform(u, ntype)

    xtrans_bysession = np.split(xtrans.squeeze(), np.array(indices)[1:-1])
    # Top FCA neurons
    subset_FCA = {}
    # Top PCA neurons
    subset_PCA = {}

    for i, session in enumerate(sessions):
        # Large component values are FBC neurons, small component values are FFC neurons
        fbc_cutoff = np.quantile(xtrans_bysession[i], quantile)
        ffc_cutoff = np.quantile(xtrans_bysession[i], 1 - quantile)

        fbc_indices = np.argwhere(xtrans_bysession[i] < fbc_cutoff).squeeze()
        ffc_indices = np.argwhere(xtrans_bysession[i] > ffc_cutoff).squeeze()

        subset_FCA[session] = fbc_indices
        subset_PCA[session] = ffc_indices

    # path = '/home/akumar/nse/neural_control/subset_indices'
    # if not os.path.exists(path):
    #     os.makedirs(path)

    # f1 = open(path + '/fca_ldasubset_q%d_%s.pkl' % (int(100 * quantile), region), 'wb')
    # f2 = open(path + '/pca_ldasubset_q%d_%s.pkl' % (int(100 * quantile), region), 'wb')

    # f1.write(pickle.dumps(subset_FCA))
    # f2.write(pickle.dumps(subset_PCA))

    # f1.close()
    # f2.close()
    
    # Color by type
    carray = cm.RdGy(range(256))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    N, bins, patches = ax.hist(xtrans, linewidth=1, bins=20)
    xbinned = np.digitize(xtrans, bins).squeeze()
    # Set each rectangle to a color gradient set by the fraction of entries that belong to each type
    fracs = []
    for i in range(len(patches)):
        x_ = np.where(xbinned == i + 1)[0]
        if len(x_) > 0:
            ntype_ = np.array(ntype)[x_]
            frac = np.sum(ntype_)/x_.size
            patches[i].set_facecolor(carray[int(255 * frac)])
            patches[i].set_edgecolor(carray[int(255*frac)])
            fracs.append(frac)
        else:
            fracs.append(np.nan)
        #patches[i].set_facecolor(carray[0])
    ax.set_xlabel('LDA Dimension 1')
    ax.set_ylabel('Count')
    ax.set_ylim(hist_ylims[region])
    ax.set_yticks(hist_yticks[region])
    #ax.set_aspect('equal')
    # Vertical colorbar

    cmap = cm.RdGy.reversed()
    norm = colors.Normalize(vmin=0, vmax=1)
    cbar= fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, orientation='vertical', label='', ticks=[0, 0.5, 1.])
    cbar.ax.set_yticklabels([])
    cbar.ax.set_ylabel('Rel. FBC Importance')
    #cbar.ax.set_yticks([0, 0.5, 1.0])

    # Add an inset - disabled for resubmission discussion with editor
    # ax.set_title('%s \n %s \n %s' % (la, deca, dra))
    # fig.tight_layout()

    axin = ax.inset_axes([0.35, 0.6, 0.4, 0.3])
    # axin.set_aspect(1)
    axin.set_ylim([0.5, 1.0])
    axin.set_xlim([0.5, 0.9])
    axin.set_yticks([0.5, 1.0])
    axin.set_xticks([0.5, 0.7, 0.9])
    axin.plot(fbc_fraction, np.mean(scores, axis=(1, 2)), color='#625487')
    axin.fill_between(fbc_fraction, np.mean(scores, axis=(1, 2)) - np.std(scores, axis=(1, 2)), np.mean(scores, axis=(1, 2)) + np.std(scores, axis=(1, 2)), alpha=0.5, color='#625487')

    if boxcox:
        figpath = PATH_DICT['figs'] + '/%s_3dlda_viz_boccox.pdf' % region
    else:
        figpath = PATH_DICT['figs'] + '/%s_3dlda_viz.pdf' % region

    fig.savefig(figpath, bbox_inches='tight', pad_inches=0)

    # UMAP raw scatter
    # Scatter in 3D - color by rfbc        
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    cmap_new = truncate_colormap(cm.RdGy_r, 0., 0.9)
    # Relative FBC/FFC score
    rfbc = np.divide(loadings_df['FCCA_loadings'].values,
                        loadings_df['FCCA_loadings'].values +\
                        loadings_df['PCA_loadings'].values)    
    rffc = np.divide(loadings_df['PCA_loadings'].values,
                        loadings_df['FCCA_loadings'].values +\
                        loadings_df['PCA_loadings'].values)    


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(u[:, 0], u[:, 0], u[:, 0],
                c=rfbc, edgecolors=(0.6, 0.6, 0.6, 0.6), 
                linewidth=0.01, s=15, cmap=cmap_new)
    ax.set_xlabel('UMAP dim. 1')
    ax.set_ylabel('UMAP dim. 2')
    ax.set_zlabel('UMAP dim. 3')
    fig.savefig('umap_scatter_%s.pdf' % region)

DIM_DICT = {
    'M1': 6,
    'S1': 3,
    'HPC_peanut':11,
    'AM': 21,
    'ML': 21,
    'VISp':10
}

if __name__ == '__main__':
    
    regions = ['M1', 'S1', 'HPC_peanut']
    for region in regions:

        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        data_path = get_data_path(region)

        loadings_df, dyn_range, act, fft = plot_feature_scatter(df, data_path,  session_key, region,  DIM_DICT[region], PATH_DICT['figs'])
        #plot_lda(df, loadings_df, dyn_range, act, fft, session_key)
        
        # run_umap_3d(df, data_path, session_key, region, dim=6)    
        # plot_umap_3d(df, session_key, region)
        
        
        print(f"Done with region: {region}")