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

sys.path.append('/home/akumar/nse/neural_control')
from utils import apply_df_filters, calc_loadings, calc_cascaded_loadings
from loaders import load_sabes, reach_segment_sabes
from scipy.ndimage import gaussian_filter1d

def get_loadings_df(dimreduc_df):

    # Load dimreduc_df and calculate loadings
    data_files = np.unique(dimreduc_df['data_file'].values)
    # Try the raw leverage scores instead
    loadings_l = []

    for i, data_file in tqdm(enumerate(data_files)):
        loadings = []
        for dimreduc_method in ['LQGCA', 'PCA']:
            loadings_fold = []
            for fold_idx in range(5):            
                df_ = apply_df_filters(dimreduc_df, data_file=data_file, fold_idx=fold_idx, dim=6, dimreduc_method=dimreduc_method)
                if dimreduc_method == 'LQGCA':
                    df_ = apply_df_filters(df_, dimreduc_args={'T': 3, 'loss_type': 'trace', 'n_init': 10})

                assert(df_.shape[0] == 1)
                V = df_.iloc[0]['coef']
                if dimreduc_method == 'PCA':
                    V = V[:, 0:6]        
                loadings_fold.append(calc_loadings(V))

            # Average loadings across folds
            loadings.append(np.mean(np.array(loadings_fold), axis=0))

        for j in range(loadings[0].size):
            d_ = {}
            d_['data_file'] = data_file
            d_['FCCA_loadings'] = loadings[0][j]
            d_['PCA_loadings'] = loadings[1][j]
            d_['nidx'] = j
            loadings_l.append(d_)                

    loadings_df = pd.DataFrame(loadings_l)
    return loadings_df

def get_loadings_and_top_neurons(dimreduc_df, n=10):

    # Load dimreduc_df and calculate loadings
    data_files = np.unique(dimreduc_df['data_file'].values)
    # Try the raw leverage scores instead
    loadings_pca = []
    idxs_pca = []
    loadings_fca = []
    idxs_fca = []

    for i, data_file in tqdm(enumerate(data_files)):
        loadings = []
        for dimreduc_method in ['LQGCA', 'PCA']:
            loadings_fold = []
            for fold_idx in range(5):            
                df_ = apply_df_filters(dimreduc_df, data_file=data_file, fold_idx=fold_idx, dim=6, dimreduc_method=dimreduc_method)
                if dimreduc_method == 'LQGCA':
                    df_ = apply_df_filters(df_, dimreduc_args={'T': 3, 'loss_type': 'trace', 'n_init': 10})
                assert(df_.shape[0] == 1)
                V = df_.iloc[0]['coef']
                if dimreduc_method == 'PCA':
                    V = V[:, 0:6]        
                loadings_fold.append(calc_loadings(V))

            if dimreduc_method == 'LQGCA':
                loadings_fca.extend(np.mean(loadings_fold, axis=0))
                idxs_fca.extend([(i, j) for j in np.arange(loadings_fold[0].size)])
            elif dimreduc_method == 'PCA':
                loadings_pca.extend(np.mean(loadings_fold, axis=0))
                idxs_pca.extend([(i, j) for j in np.arange(loadings_fold[0].size)])
   
    # Assemble the disjoint sets across all recording sessions.
    # Want to end up with n * n_data_files neurons, but pooled across sessions now.
    top_neurons_pca = []
    top_neurons_fca = []
    
    N = n * len(data_files)

    pca_ordering = np.argsort(loadings_pca)[::-1]
    fca_ordering = np.argsort(loadings_fca)[::-1]

    def empty_min(x):
        if len(x) == 0:
            return np.inf
        else:
            return np.min(x)

    idx = 0
    while not np.all([len(x) >= N for x in [top_neurons_pca, top_neurons_fca]]):
        top_pca_candidate = idxs_pca[pca_ordering[idx]]

        # Accept the candidate only if it is not contained in the top N FCCA neurons
        candidate_fca_idx = idxs_fca.index(top_pca_candidate)

        if not candidate_fca_idx in fca_ordering[:N]:
            top_neurons_pca.append(pca_ordering[idx])

        top_fca_candidate = idxs_fca[fca_ordering[idx]]

        candidate_pca_idx = idxs_pca.index(top_fca_candidate)

        if not candidate_pca_idx in pca_ordering[:N]:
            top_neurons_fca.append(fca_ordering[idx])
    
        idx += 1

    return top_neurons_pca, top_neurons_fca, loadings_pca, loadings_fca

# Calculate the top FFC/FBC neurons and return their indices for each 
# data file for use in loading
def subset_selection(dimreduc_df, quantile, M1=True):
    loadings_df = get_loadings_df(dimreduc_df)
    keys = ['FCCA_loadings', 'PCA_loadings']

    # Top FCA neurons
    subset_FCA = {}
    # Top PCA neurons
    subset_PCA = {}
    data_files = np.unique(dimreduc_df['data_file'].values)
    for i, data_file in enumerate(data_files):
        # Per recording session
        yf_ = []
        yp_ = []
        for j in range(2):
            df = apply_df_filters(loadings_df, data_file=data_files[i])
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

        subset_FCA[data_file] = fca_indices
        subset_PCA[data_file] = pca_indices

    # Save away
    path = '/home/akumar/nse/neural_control/subset_indices'
    if not os.path.exists(path):
        os.makedirs(path)

    if M1:
        f1 = open(path + '/fca_subset_q%d_M1.pkl' % int(100 * quantile), 'wb')
        f2 = open(path + '/pca_subset_q%d_M1.pkl' % int(100 * quantile), 'wb')
    else:
        f1 = open(path + '/fca_subset_q%d_S1.pkl' % int(100 * quantile), 'wb')
        f2 = open(path + '/pca_subset_q%d_S1.pkl' % int(100 * quantile), 'wb')

    f1.write(pickle.dumps(subset_FCA))
    f2.write(pickle.dumps(subset_PCA))

    f1.close()
    f2.close()

def make_scatter(dimreduc_df, figpath):
    top_neurons_pca, top_neurons_fca, loadings_pca, loadings_fca = get_loadings_and_top_neurons(dimreduc_df, n=10)

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

    ax.set_xlim([-6, 0.1])
    ax.set_ylim([-6, 0.1])
    ax.set_xticks([0, -3, -6])
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

def run_lda_analysis(dimreduc_df, data_path, M1=True, overwrite=False):

    if M1:
        savepath = 'psth_clustering_tmpM1.pkl'
        region = 'M1'
    else:
        savepath = 'psth_clustering_tmpS1.pkl'
        region='S1'
    if not os.path.exists('tmp/' + savepath) or overwrite:

        xall = []
        print('Collecting PSTH')
        loadings_df = get_loadings_df(dimreduc_df)
        data_files = np.unique(loadings_df['data_file'].values)
        for h, data_file in enumerate(data_files):
            dat = load_sabes('%s/%s' % (data_path, data_file), boxcox=None, high_pass=False, region=region)
            dat_segment = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])
            T = 30
            t = np.array([t_[1] - t_[0] for t_ in dat_segment['transition_times']])
            valid_transitions = np.arange(t.size)[t >= T]

            # (Bin size 50 ms)
            time = 50 * np.arange(T)        
            # Store trajectories for subsequent pairwise analysis
            n = dat['spike_rates'].shape[-1]
            x = np.zeros((n, time.size))
            for j in range(n):
                x_ = np.array([dat['spike_rates'][0, dat_segment['transition_times'][idx][0]:dat_segment['transition_times'][idx][0] + T, j] 
                            for idx in valid_transitions])
                #x_ = StandardScaler().fit_transform(x_.T).T
                x_ = gaussian_filter1d(x_, sigma=2)
                x_ = np.mean(x_, axis=0)
                x[j, :] = x_
            xall.append(x)
        xall_stacked = np.vstack(xall)

        param = (4, 0.2, 50)
        # Share the UMAP projection across recording sessions - measure classificationa ccuraacy per recording session
        fit = umap.UMAP(min_dist=param[1], n_neighbors=param[2], n_components=param[0])
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
                    df = apply_df_filters(loadings_df, data_file=data_files[i])
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

        if not os.path.exists('tmp'):
            os.makedirs('tmp')

        with open('tmp/' + savepath, 'wb') as f:
            f.write(pickle.dumps(scores))
            f.write(pickle.dumps(dummy_scores))
            f.write(pickle.dumps(random_scores))
            f.write(pickle.dumps(loadings_df))
            f.write(pickle.dumps(xall))
            f.write(pickle.dumps(u))
            # f.write(pickle.dumps(class_sizes))
        
# quantile - save away indices of subset of neurons with high/low LDA component value
def plot_lda_analysis(dimreduc_df, M1=True, quantile=0.75):

    if M1:
        savepath = 'psth_clustering_tmpM1.pkl'
    else:
        savepath = 'psth_clustering_tmpS1.pkl'

    if not os.path.exists('tmp/' + savepath):
        raise ValueError('Call run_lda_analysis first')

    with open('tmp/' + savepath, 'rb') as f:
        scores = pickle.load(f)
        dummy_scores = pickle.load(f)
        random_scores = pickle.load(f)
        loadings_df = pickle.load(f)
        xall = pickle.load(f)
        u = pickle.load(f)

    data_files = np.unique(loadings_df['data_file'])

    fbc_fraction = np.linspace(0.5, 0.95, 25)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.plot(fbc_fraction, np.mean(scores, axis=(1, 2)))
    ax.fill_between(fbc_fraction, np.mean(scores, axis=(1, 2)) - np.std(scores, axis=(1, 2)), np.mean(scores, axis=(1, 2)) + np.std(scores, axis=(1, 2)), alpha=0.5)

    ax.plot(fbc_fraction, np.mean(dummy_scores, axis=(1, 2)))
    ax.fill_between(fbc_fraction, np.mean(dummy_scores, axis=(1, 2)) - np.std(dummy_scores, axis=(1, 2)), np.mean(dummy_scores, axis=(1, 2)) + np.std(dummy_scores, axis=(1, 2)), alpha=0.5)

    ax.plot(fbc_fraction, np.mean(random_scores, axis=(1, 2, 3)), color='purple')
    ax.fill_between(fbc_fraction, np.mean(random_scores, axis=(1, 2, 3)) - np.std(random_scores, axis=(1, 2, 3)), 
                    np.mean(random_scores, axis=(1, 2, 3)) + np.std(random_scores, axis=(1, 2, 3)), alpha=0.5, color='purple')

    ax.legend(['Data', 'Dummy', 'Random'], loc='lower right')
    ax.set_xlabel('FBC Quantile', fontsize=14)
    ax.set_ylabel('Classification Accuracy', fontsize=14)

    # ax[1].plot(fbc_fraction, np.mean(class_sizes[:, :, 0], axis=1), color='r')
    # ax[1].plot(fbc_fraction, np.mean(class_sizes[:, :, 1], axis=1), color='k')
    # ax[1].set_xlabel('FBC Quantile', fontsize=14)
    # ax[1].set_ylabel('Average Class Size', fontsize=14)
    # fig.tight_layout()
    if M1:
        fig.savefig('/home/akumar/nse/neural_control/figs/loco_indy_merge/umap_clusteringLDAM1.pdf', bbox_inches='tight', pad_inches=1)
    else:
        fig.savefig('/home/akumar/nse/neural_control/figs/loco_indy_merge/umap_clusteringLDAM1.pdf', bbox_inches='tight', pad_inches=1)

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
            df = apply_df_filters(loadings_df, data_file=data_files[i])
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

    for i, data_file in enumerate(data_files):
        # Large component values are FBC neurons, small component values are FFC neurons
        fbc_cutoff = np.quantile(xtrans_bysession[i], quantile)
        ffc_cutoff = np.quantile(xtrans_bysession[i], 1 - quantile)

        fbc_indices = np.argwhere(xtrans_bysession[i] < fbc_cutoff).squeeze()
        ffc_indices = np.argwhere(xtrans_bysession[i] > ffc_cutoff).squeeze()

        subset_FCA[data_file] = fbc_indices
        subset_PCA[data_file] = ffc_indices    

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
    ax.set_yticks([0, 150, 300])
    #ax.set_aspect('equal')
    # Vertical colorbar

    cmap = cm.RdGy.reversed()
    norm = colors.Normalize(vmin=0, vmax=1)
    cbar= fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, orientation='vertical', label='', ticks=[0, 0.5, 1.])
    cbar.ax.set_yticklabels([])

    #cbar.ax.set_yticks([0, 0.5, 1.0])

    # Add an inset

    axin = ax.inset_axes([0.35, 0.6, 0.4, 0.3])
    # axin.set_aspect(1)
    axin.set_ylim([0.5, 1.0])
    axin.set_xlim([0.5, 0.9])
    axin.set_yticks([0.5, 1.0])
    axin.set_xticks([0.5, 0.7, 0.9])
    axin.plot(fbc_fraction, np.mean(scores, axis=(1, 2)), color='#625487')
    axin.fill_between(fbc_fraction, np.mean(scores, axis=(1, 2)) - np.std(scores, axis=(1, 2)), np.mean(scores, axis=(1, 2)) + np.std(scores, axis=(1, 2)), alpha=0.5, color='#625487')
    #fig.savefig('/home/akumar/nse/neural_control/figs/loco_indy_merge/m1_lda_viz.pdf', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    M1 = True
    subset_index = 0
    if M1:
        with open('/mnt/Secondary/data/postprocessed/sabes_M1subset2_dec_df.pkl', 'rb') as f:
            rl = pickle.load(f)
    else:
        with open('/mnt/Secondary/data/postprocessed/sabes_S1subset2_dec_df.pkl', 'rb') as f:
            rl = pickle.load(f)

    df = pd.DataFrame(rl)

    # Pick one
    decoder_arg = df.iloc[0]['decoder_args']
    df = apply_df_filters(df, decoder_args=decoder_arg)

    if M1:
        make_scatter(df, '/home/akumar/nse/neural_control/figs/revisions/IS_scatter_subset%d.pdf' % subset_index) 
        # run_lda_analysis(df, '/mnt/Secondary/data/sabes', M1)
        # plot_lda_analysis(df, M1)
    else:
        # Further trim the df by a single decoder arg
        make_scatter(df, '/home/akumar/nse/neural_control/figs/revisions/IS_scatterS1_subset%d.pdf' % subset_index) 
        # run_lda_analysis(df, '/mnt/Secondary/data/sabes', M1)
        # plot_lda_analysis(df, M1)