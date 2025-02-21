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

import matplotlib.cm as cm
import matplotlib.colors as colors

from region_select import *
from config import PATH_DICT
sys.path.append(PATH_DICT['repo'])
from utils import calc_loadings, calc_cascaded_loadings


scatter_lims = {
    'M1': [-6.2, 0.1],
    'S1': [-6.2, 0.1],
    'M1_trialized': [-6.5, 0.1],
    'M1_maze': [-2.5, 0.1],
    'HPC_peanut': [-6, 0.1],
    'AM': [-6, 0.1],
    'ML': [-6, 0.1],
    'mPFC': [-6, 0.1],
    'HPC': [-5.5, 0.1],
    'VISp': [-6, 0.1]
}

scatter_ticks = {
    'M1': [0, -3 ,-6],
    'S1': [0, -3, -6],
    'M1_trialized': [0, -3, -6],
    'M1_maze': [0, -1, -2],
    'HPC_peanut': [0, -3, -6],
    'AM': [0, -3, -6],
    'ML': [0, -3, -6],
    'mPFC': [0, -3, -6],
    'HPC': [0, -2.5, -5],
    'VISp': [0, -3, -6]
}


hist_ylims = {
    'M1': [0, 300],
    'S1': [0, 120],
    'M1_trialized': [0, 300],
    'M1_maze': [0, 40],
    'HPC_peanut': [0, 30],
    'AM': [0, 80],
    'ML': [0, 110],
    'mPFC': [0, 10],
    'HPC': [0, 10],
    'VISp': [0,155]
}

hist_yticks = {
    'M1': [0, 150, 300],
    'S1': [0, 60, 120],
    'M1_trialized': [0, 150, 300],
    'M1_maze': [0, 20, 40],
    'HPC_peanut': [0, 15, 30],
    'AM': [0, 40, 80],
    'ML': [0, 50, 100],
    'mPFC': [0, 5, 10],
    'HPC': [0, 5, 10],
    'VISp': [0, 50, 100, 150]

}

def get_loadings_df(dimreduc_df, session_key, dim=6):
        
    sessions = np.unique(dimreduc_df[session_key].values)
    # Try the raw leverage scores instead
    loadings_l = []

    for i, session in tqdm(enumerate(sessions)):
        loadings = []
        for dimreduc_method in [['LQGCA', 'FCCA'], 'PCA']:
            loadings_fold = []
            for fold_idx in range(5):            
                df_filter = {session_key:session, 'fold_idx':fold_idx, 'dim':dim, 'dimreduc_method':dimreduc_method}
                df_ = apply_df_filters(dimreduc_df, **df_filter)
                try:
                    assert(df_.shape[0] == 1)
                except:
                    pdb.set_trace()
                V = df_.iloc[0]['coef']
                if dimreduc_method == 'PCA':
                    V = V[:, 0:dim]        
                loadings_fold.append(calc_loadings(V))

            # Average loadings across folds
            loadings.append(np.mean(np.array(loadings_fold), axis=0))

        for j in range(loadings[0].size):
            d_ = {}
            d_[session_key] = session
            d_['FCCA_loadings'] = loadings[0][j]
            d_['PCA_loadings'] = loadings[1][j]
            d_['nidx'] = j
            loadings_l.append(d_)                

    loadings_df = pd.DataFrame(loadings_l)
    return loadings_df

def get_loadings_and_top_neurons(dimreduc_df, session_key, dim=6, n=10):

    # Load dimreduc_df and calculate loadings
    sessions = np.unique(dimreduc_df[session_key].values)
    # Try the raw leverage scores instead
    loadings_pca = []
    idxs_pca = []
    loadings_fca = []
    idxs_fca = []

    for i, session in tqdm(enumerate(sessions)):
        loadings = []
        for dimreduc_method in [['LQGCA', 'FCCA'], 'PCA']:
            loadings_fold = []
            for fold_idx in range(5):            
                df_filter = {session_key:session, 'fold_idx':fold_idx, 'dim':dim, 'dimreduc_method':dimreduc_method}
                df_ = apply_df_filters(dimreduc_df, **df_filter)
                assert(df_.shape[0] == 1)
                V = df_.iloc[0]['coef']
                if dimreduc_method == 'PCA':
                    V = V[:, 0:dim]        
                loadings_fold.append(calc_loadings(V))

            if dimreduc_method == 'PCA':
                loadings_pca.extend(np.mean(loadings_fold, axis=0))
                idxs_pca.extend([(i, j) for j in np.arange(loadings_fold[0].size)])
            else:
                loadings_fca.extend(np.mean(loadings_fold, axis=0))
                idxs_fca.extend([(i, j) for j in np.arange(loadings_fold[0].size)])

    return loadings_pca, loadings_fca

# Calculate the top FFC/FBC neurons and return their indices for each 
# data file for use in loading
def subset_selection(dimreduc_df, quantile, session_key, region='M1'):
    loadings_df = get_loadings_df(dimreduc_df)
    keys = ['FCCA_loadings', 'PCA_loadings']

    # Top FCA neurons
    subset_FCA = {}
    # Top PCA neurons
    subset_PCA = {}
    sessions = np.unique(dimreduc_df[session_key].values)
    for i, session in enumerate(sessions):
        # Per recording session
        yf_ = []
        yp_ = []
        for j in range(2):
            df = apply_df_filters(loadings_df,**{session_key:session})
            x1 = df[keys[j]].values
            if j == 0:
                yf_.append(x1)
            else:
                yp_.append(x1)

        rfbc = yf_[-1]/(yf_[-1] + yp_[-1])
        rffc = yp_[-1]/(yf_[-1] + yp_[-1])

        fca_cutoff = np.quantile(rfbc, quantile)
        pca_cutoff = np.quantile(rffc, quantile)

        fca_indices = np.argwhere(rfbc < fca_cutoff).squeeze()
        pca_indices = np.argwhere(rffc < pca_cutoff).squeeze()

        subset_FCA[session] = fca_indices
        subset_PCA[session] = pca_indices

    # Save away
    path = '/home/akumar/nse/neural_control/subset_indices'
    if not os.path.exists(path):
        os.makedirs(path)

    f1 = open(path + '/fca_subset_q%d_%s.pkl' % (int(100 * quantile), region), 'wb')
    f2 = open(path + '/pca_subset_q%d_%s.pkl' % (int(100 * quantile), region), 'wb')

    f1.write(pickle.dumps(subset_FCA))
    f2.write(pickle.dumps(subset_PCA))

    f1.close()
    f2.close()

def make_scatter(dimreduc_df, session_key, region, dim):

    if region in ['AM', 'ML']:  
        dimreduc_df_sess = apply_df_filters(dimreduc_df, **{'loader_args':{'region': region}})
        loadings_pca, loadings_fca = get_loadings_and_top_neurons(dimreduc_df_sess, session_key, dim=dim, n=10)
    else:
        loadings_pca, loadings_fca = get_loadings_and_top_neurons(dimreduc_df, session_key, dim=dim, n=10)

    figpath = PATH_DICT['figs'] + '/IS_scatter%s.pdf' % region
    fig, ax = plt.subplots(figsize=(5, 5))

    x1 = np.array(loadings_fca)
    x2 = np.array(loadings_pca)
        
    def drawPieMarker(xs, ys, ratios, colors):
        assert sum(ratios) <= 1 + 1e-3, 'sum of ratios needs to be < 1'
        markers = []
        previous = 0
        # calculate the points of the pie pieces
        for color, ratio in zip(colors, ratios):
            this = 2 * np.pi * ratio + previous
            x  = [0] + np.cos(np.linspace(previous, this, 10)).tolist() + [0]
            y  = [0] + np.sin(np.linspace(previous, this, 10)).tolist() + [0]
            xy = np.column_stack([x, y])
            previous = this
            markers.append({'marker':xy, 's':10, 'facecolor':color})

        # scatter each of the pie pieces to create pies
        for marker in markers:
            ax.scatter(xs, ys, **marker, alpha=0.35)

    # for x1_, x2_ in tqdm(zip(x1, x2)):
    #     drawPieMarker([np.log10(x1_)], [np.log10(x2_)], [x1_/(x1_ + x2_), x2_/(x1_ + x2_)], ['red', 'black'])

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    cmap_new = truncate_colormap(cm.RdGy_r, 0., 0.9)
    ratio = np.divide(x1, x1 + x2)

    h = ax.scatter(np.log10(x1), np.log10(x2), c=ratio, edgecolors=(0.6, 0.6, 0.6, 0.6), linewidth=0.01, s=15, cmap=cmap_new)
    # Highlight top neurons by modulating size and linewidth
    # # s = 10 * np.zeros(x1.shape)
    # # s[top_neurons_fca] = 25
    # # s[top_neurons_pca] = 25
    # h = ax.scatter(np.log10(x1), np.log10(x2), c=ratio, edgecolors=(0.8, 0.8, 0.8, 0.8), linewidth=0.01, s=s, cmap=cmap_new)
    cbar = plt.colorbar(h, cax=fig.add_axes([0.925, 0.25, 0.025, 0.5]))
    cbar.set_label('Relative FBC Importance', fontsize=16)
    cbar.set_ticks([0.2, 0.8])
    cbar.ax.tick_params(labelsize=16)
    # Annotate with the spearman-r
    r = scipy.stats.spearmanr(x1, x2)
    print('Spearman sample size:%d' % x1.size)
    print('Spearman:%f' % r[0])
    print('Spearman p:%f' % r[1])
    
    #ax.set_xlim(scatter_lims[region])
    #ax.set_ylim(scatter_lims[region])
    #ax.set_xticks(scatter_ticks[region])
    #ax.set_yticks(scatter_ticks[region])

    ax.set_xlim([-2.5, 0.1])
    ax.set_xticks([0, -1, -2])
    
    ax.set_ylim([-6, 0.1])
    ax.set_yticks([0, -3, -6])
    ax.set_xlabel('Log FBC Importance Score', fontsize=18)
    ax.set_ylabel('Log FFC Importance Score', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)

    # What is the spearman correlation in the intersection of the upper quartiles?
    # idxs1 = np.argwhere(x1 > q1_fca)[:, 0]
    # idxs2 = np.argwhere(x2 > q1_pca)[:, 0]
    # intersct = np.array(list(set(idxs1).intersection(set(idxs2))))

    #ax.annotate('Spearman ' + r'$\rho$' +'=%.2f' % r, (-4.8, -4.5), fontsize=16)
    # ax.annotate('Upper-quartile r=%.2f' % r2, (-4.8, -0.5), fontsize=14)
    fig.savefig(figpath, bbox_inches='tight', pad_inches=0)
    return r

def run_lda_analysis(dimreduc_df, data_path, session_key, region, dim, boxcox=False,
                     overwrite=False):
    if boxcox:
        savepath = '/psth_clustering_tmp%s_boxcox.pkl' % region
    else:
        savepath = '/psth_clustering_tmp%s.pkl' % region

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
                dimreduc_df_sess = apply_df_filters(dimreduc_df, **{session_key:session, 'loader_args':{'region': region}})
                x = get_rates_smoothed(data_path, region, session, 
                                      full_arg_tuple=dimreduc_df_sess['full_arg_tuple'], std=False)
            else:
                x = get_rates_smoothed(data_path, region, session,
                              loader_args=dimreduc_df.iloc[0]['loader_args'], std=False)
            xall.append(x)

        # MIH HACK TO CHANGE LATER !!!
        if region in ['AM', 'ML']:
            xall = [x[:, :19] for x in xall]

        xall_stacked = np.vstack(xall)
        param = (4, 0.2, 50)
        # Share the UMAP projection across recording sessions - measure classificationa ccuraacy per recording session
        if region == 'AM':
            fit = umap.UMAP(min_dist=param[1], n_neighbors=param[2], n_components=param[0], random_state=None, transform_seed=None)
        else:
            fit = umap.UMAP(min_dist=param[1], n_neighbors=param[2], n_components=param[0], random_state=42, transform_seed=42)
        print('Fiting UMAP')
        u = fit.fit_transform(xall_stacked)        
        fbc_fraction = np.linspace(0.5, 0.95, 25)
        ncv = 5
        nrandom = 100
        # Partition by data file
        scores = np.zeros((len(fbc_fraction), len(xall), ncv))
        dummy_scores = np.zeros((len(fbc_fraction), len(xall), ncv))
        random_scores = np.zeros((len(fbc_fraction), len(xall), ncv, nrandom))
        keys = ['FCCA_loadings', 'PCA_loadings']
        indices = list(np.cumsum([len(x_) for x_ in xall]))
        indices.insert(0, 0)

        for ii, fbcf in tqdm(enumerate(fbc_fraction)):
            for i in range(len(xall)):

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

                # get the
                u_i = u[indices[i]:indices[i+1], :]
                #logreg = LogisticRegression()
                lda = LinearDiscriminantAnalysis(n_components=1)
                # perform 10-fold cross-validation
                scores[ii, i] = cross_val_score(lda, u_i, ntype, cv=ncv)
                dummy_scores[ii, i] = cross_val_score(DummyClassifier(strategy='stratified'), u_i, ntype, cv=ncv)

                # Compare also to 100 random assignments of neuron types
                for k in range(nrandom):
                    ntype_rand = np.random.permutation(ntype)
                    random_scores[ii, i, :, k] = cross_val_score(lda, u_i, ntype_rand, cv=ncv)

        with open(PATH_DICT['tmp'] + savepath, 'wb') as f:
            f.write(pickle.dumps(scores))
            f.write(pickle.dumps(dummy_scores))
            f.write(pickle.dumps(random_scores))
            f.write(pickle.dumps(loadings_df))
            f.write(pickle.dumps(xall))
            f.write(pickle.dumps(u))
            # f.write(pickle.dumps(class_sizes))
        
# quantile - save away indices of subset of neurons with high/low LDA component value
def plot_lda_analysis(dimreduc_df, session_key, region='M1', quantile=0.75, boxcox=False):

    if boxcox:
        savepath = '/psth_clustering_tmp%s_boxcox.pkl' % region
    else:
        savepath = '/psth_clustering_tmp%s.pkl' % region
    if not os.path.exists(PATH_DICT['tmp'] + savepath):
        raise ValueError('Call run_lda_analysis first')

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
    ax.fill_between(fbc_fraction, np.mean(scores, axis=(1, 2)) - np.std(scores, axis=(1, 2)), np.mean(scores, axis=(1, 2)) + np.std(scores, axis=(1, 2)), alpha=0.5, label='_nolegend_')

    ax.plot(fbc_fraction, np.mean(dummy_scores, axis=(1, 2)))
    ax.fill_between(fbc_fraction, np.mean(dummy_scores, axis=(1, 2)) - np.std(dummy_scores, axis=(1, 2)), np.mean(dummy_scores, axis=(1, 2)) + np.std(dummy_scores, axis=(1, 2)), alpha=0.5, label='_nolegend_')

    ax.plot(fbc_fraction, np.mean(random_scores, axis=(1, 2, 3)), color='purple')
    ax.fill_between(fbc_fraction, np.mean(random_scores, axis=(1, 2, 3)) - np.std(random_scores, axis=(1, 2, 3)), 
                    np.mean(random_scores, axis=(1, 2, 3)) + np.std(random_scores, axis=(1, 2, 3)), alpha=0.5, color='purple', label='_nolegend_')

    ax.legend(['Data', 'Dummy', 'Random'], loc='lower right')
    ax.set_xlabel('FBC Quantile', fontsize=14)
    ax.set_ylabel('Classification Accuracy', fontsize=14)
    ax.set_ylim([0.5, 1])

    # Statistical test on the difference between Data and Dummy classification)
    stat, p = scipy.stats.wilcoxon(np.mean(scores, axis=2)[0], np.mean(dummy_scores, axis=2)[0], alternative='greater')
    print(f'Avg cross-validated score at 0.5: {np.mean(scores, axis=(1, 2))[0]}')
    print(f'Score vs. dummy at 0.5: p={p}')

    # ax[1].plot(fbc_fraction, np.mean(class_sizes[:, :, 0], axis=1), color='r')
    # ax[1].plot(fbc_fraction, np.mean(class_sizes[:, :, 1], axis=1), color='k')
    # ax[1].set_xlabel('FBC Quantile', fontsize=14)
    # ax[1].set_ylabel('Average Class Size', fontsize=14)
    # fig.tight_layout()
    if boxcox:
        figpath = PATH_DICT['figs'] + '/umap_clusteringLDA%s_boxcox.pdf' % region
    else:
        figpath = PATH_DICT['figs'] + '/umap_clusteringLDA%s.pdf' % region

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

    axin = ax.inset_axes([0.35, 0.6, 0.4, 0.3])
    # axin.set_aspect(1)
    axin.set_ylim([0.5, 1.0])
    axin.set_xlim([0.5, 0.9])
    axin.set_yticks([0.5, 1.0])
    axin.set_xticks([0.5, 0.7, 0.9])
    axin.plot(fbc_fraction, np.mean(scores, axis=(1, 2)), color='#625487')
    axin.fill_between(fbc_fraction, np.mean(scores, axis=(1, 2)) - np.std(scores, axis=(1, 2)), np.mean(scores, axis=(1, 2)) + np.std(scores, axis=(1, 2)), alpha=0.5, color='#625487')

    if boxcox:
        figpath = PATH_DICT['figs'] + '/%s_lda_viz_boccox.pdf' % region
    else:
        figpath = PATH_DICT['figs'] + '/%s_lda_viz.pdf' % region

    fig.savefig(figpath, bbox_inches='tight', pad_inches=0)
    

def make_PSTHs(df, session_key, data_path, region):

    # Get raw rates
    session = np.unique(df[session_key].values)[0]
        
    dimreduc_df_sess = apply_df_filters(df, **{session_key:session, 'loader_args':{'region': region}})
    binWidth = dimreduc_df_sess['loader_args'][0]['bin_width']
    
    if region in ['ML', 'AM']:
        spike_rates, spike_times = get_tsao_raw_rates(data_path, region, session, full_arg_tuple=dimreduc_df_sess['full_arg_tuple'])
    else:

        unique_loader_args = list({frozenset(d.items()) for d in df['loader_args']})
        loader_args=dict(unique_loader_args[loader_kwargs[region]['load_idx']])
        
        spike_rates = get_rates_raw(data_path, region, session, loader_args=loader_args)

    
    #rates = get_rates_raw(data_path, region, session, full_arg_tuple=dimreduc_df_sess['full_arg_tuple'])
    #rates = get_rates_smoothed(data_path, region, session, full_arg_tuple=dimreduc_df_sess['full_arg_tuple'])
    loadings_df_all = get_loadings_df(df, session_key, dim=dim_dict[region])    
    loadings_df = apply_df_filters(loadings_df_all, **{'data_file':session})
                
    
    nUnits = 20
    #fcca_sort = np.argsort(loadings_df['FCCA_loadings'])[::-1]
    fcca_sort = np.argsort(loadings_df['FCCA_loadings'] / (loadings_df['FCCA_loadings']+loadings_df['PCA_loadings']))[::-1]
    fcca_inds = loadings_df['nidx'][fcca_sort[:nUnits]]
    
    
    #pca_sort = np.argsort(loadings_df['PCA_loadings'])[::-1]
    pca_sort = np.argsort(loadings_df['PCA_loadings'] / (loadings_df['FCCA_loadings']+loadings_df['PCA_loadings']))[::-1]
    pca_inds = loadings_df['nidx'][pca_sort[:nUnits]]
    
    #time_cut_ind = int(550 // binWidth)
    filter_size = 1
    fcca_avg_rates = np.mean(gaussian_filter1d(spike_rates[:, :, fcca_inds], sigma=filter_size, axis=1), 0).squeeze()
    pca_avg_rates = np.mean(gaussian_filter1d(spike_rates[:, :, pca_inds], sigma=filter_size, axis=1), 0).squeeze()
    
    if region == 'ML':
        column_max_values = np.max(fcca_avg_rates, axis=0)
        two_largest_indices = np.argsort(column_max_values)[-2:]  # Get the last two indices of sorted values
        fcca_avg_rates = fcca_avg_rates[:, np.setdiff1d(np.arange(fcca_avg_rates.shape[1]), two_largest_indices)]


    time = np.arange(np.shape(spike_rates)[1])
    time_labels = time*binWidth

    
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    ax[0].plot(time, fcca_avg_rates, color="red")
    ax[0].set_title(f"Top {nUnits} FCCA Unit PSTHs")
    ax[0].set_ylabel("Firing Rate")
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_xticks(time[::2])
    ax[0].set_xticklabels(time_labels[::2])
    
    ax[1].plot(time, pca_avg_rates, color="black")
    ax[1].set_title(f"Top {nUnits} PCA Unit PSTHs")
    ax[1].set_ylabel("Firing Rate")
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_xticks(time[::2])
    ax[1].set_xticklabels(time_labels[::2])
    
    plt.subplots_adjust(hspace=0.5)
    figpath = PATH_DICT['figs'] + f"/PSTHs_{region}.pdf"
    fig.savefig(figpath, bbox_inches='tight', pad_inches=0)
    
    """
    for uID in range(nUnits):
        
        unitID = fcca_inds[fcca_inds.keys()[uID]]
        
        num_trials = np.shape(spike_times)[0]
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        for trial in range(num_trials):
            spikes_for_trial = spike_times[trial][unitID]  # Spike times for the current trial and unit
            
            # Plot each spike as an 'x'
            ax.scatter(spikes_for_trial, np.full_like(spikes_for_trial, trial + 1), color="black", marker='x')

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Trials")
        ax.set_title(f"Raster Plot for FCCA Unit {uID}")
        ax.invert_yaxis()  # Invert y-axis to have trials from top to bottom

        figpath = PATH_DICT['figs'] + f"/Rasters/fcca_rasters_{region}_unit{uID}.pdf"
        fig.savefig(figpath, bbox_inches='tight', pad_inches=0)      
        
    for uID in range(nUnits):
        
        unitID = pca_inds[pca_inds.keys()[uID]]
        
        num_trials = np.shape(spike_times)[0]
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        for trial in range(num_trials):
            spikes_for_trial = spike_times[trial][unitID]  # Spike times for the current trial and unit
            
            # Plot each spike as an 'x'
            ax.scatter(spikes_for_trial, np.full_like(spikes_for_trial, trial + 1), color="black", marker='x')

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Trials")
        ax.set_title(f"Raster Plot for PCA Unit {uID}")
        ax.invert_yaxis()  # Invert y-axis to have trials from top to bottom

        figpath = PATH_DICT['figs'] + f"/Rasters/pca_rasters_{region}_unit{uID}.pdf"
        fig.savefig(figpath, bbox_inches='tight', pad_inches=0)     
            """
        

dim_dict = {
    'M1': 6,
    'M1_trialized':6,
    'S1': 6,
    'HPC_peanut': 11,
    'M1_maze': 6,
    'AM': 21,
    'ML': 21,
    'mPFC':15,
    'HPC': 15,
    'VISp':10
}

if __name__ == '__main__':
    #regions = ['M1', 'S1', 'HPC_peanut', 'M1_maze', 'M1_trialized', 'AM', 'ML', 'HPC', 'mPFC']
    
    # Note: HPC_peanut's decode file in clustersfs/FCCA/postprocessed is not concatenated correctly (doesn't have dimreduc info in it)
    #regions = ['M1', 'S1', 'M1_maze', 'M1_trialized', 'AM', 'ML', 'HPC', 'mPFC']

    regions = ['AM', 'ML'] 
    
    
    overwrite = True # Re-run UMAP/LDA?
    boxcox = [False] # Should we apply a boxcox transformation upon loadings
    
    for region in tqdm(regions):
        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        data_path = get_data_path(region)

        make_PSTHs(df, session_key, data_path, region)
        #continue
        
        _ = make_scatter(df, session_key, region, dim_dict[region])
        for bc in boxcox:
            # # for q in np.array([0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
            #     subset_selection(df, q, session_key, region)
            # run_lda_analysis(df, data_path, session_key, region, 
            #                  dim=dim_dict[region], boxcox=bc, overwrite=overwrite)
            plot_lda_analysis(df, session_key, region, boxcox=bc)
            
        print(f"Done with: {region}")
