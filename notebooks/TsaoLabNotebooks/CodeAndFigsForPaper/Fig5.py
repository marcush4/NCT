import pdb
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy 
import pickle
import pandas as pd

from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

sys.path.append('/home/marcush/projects/neural_control/')
sys.path.append('/home/marcush/projects/neural_control/analysis_scripts/')
sys.path.append('/home/marcush/projects/github_repos')

from utils import apply_df_filters, calc_loadings
from psth_ilmerge import get_top_neurons
from region_select import *



def get_su_calcs(region):
    # Fill in directories for su calcs data:

    if region in ['AM', 'ML']:

        su_calcs_path = f'/clusterfs/NSDS_data/FCCA/postprocessed/tsao_su_calcs_{region}.pkl'     
        with open(su_calcs_path, 'rb') as f:
            tsao_su_stats = pickle.load(f)

        su_calcs_df = pd.DataFrame(tsao_su_stats)


    return su_calcs_df

def get_marginal_dfs(region):
    # Fill in directories for marginals:

    if region in ['AM', 'ML']:

        marginals_path = '/clusterfs/NSDS_data/FCCA/postprocessed/tsao_marginal_decode_df.pkl'
        with open(marginals_path, 'rb') as f:
            rl = pickle.load(f)
        marginals_df = pd.DataFrame(rl)


    return marginals_df    


def get_scalar(df_, stat, neu_idx):

    if stat == 'decoding_weights':
        if 'decoding_window' in df_.iloc[0]['decoder_params'].keys():
            decoding_win = df_.iloc[0]['decoder_params']['decoding_window']
            c = calc_loadings(df_.iloc[0]['decoding_weights'][2:4].T, d=decoding_win)[neu_idx]
        else:
            c = calc_loadings(df_.iloc[0]['decoding_weights'][2:4].T)[neu_idx]

    elif stat == 'encoding_weights':
        if 'decoding_window' in df_.iloc[0]['decoder_params'].keys():
            decoding_win = df_.iloc[0]['decoder_params']['decoding_window']
            c =  calc_loadings(df_.iloc[0]['encoding_weights'], d=decoding_win)[neu_idx]    
        else:
            c =  calc_loadings(df_.iloc[0]['encoding_weights'])[neu_idx]    

    elif stat in ['su_r2_pos', 'su_r2_vel', 'su_r2_enc', 'su_var', 'su_act', 'su_decoding_r2', 'su_encoding_r2']:
        c = df_.iloc[0][stat][neu_idx]  

    elif stat == 'orientation_tuning':
        c = np.zeros(8)
        for j in range(8):
            c[j] = df_.loc[df_['bin_idx'] == j].iloc[0]['tuning_r2'][j, 2, neu_idx]
        c = np.mean(c)
        # c = odf_.iloc[0]

    return c

def make_plot_1(yp_pred, yf_pred, yp_, yf_, r1p_, r1f_, coefp, coeff, i, figpath, region):
        
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].scatter(yp_pred.squeeze(), np.array(yp_).squeeze(), alpha=0.5)
        ax[1].scatter(yf_pred.squeeze(), np.array(yf_).squeeze(), alpha=0.5)
        ax[0].set_title('Spearman: %f, Pearson: %f' % (r1p_, scipy.stats.pearsonr(yp_pred.squeeze(), np.array(yp_).squeeze())[0]), fontsize=10)
        ax[1].set_title('Spearman: %f, Pearson: %f' % (r1f_, scipy.stats.pearsonr(yf_pred.squeeze(), np.array(yf_).squeeze())[0]), fontsize=10)

        ax[0].set_xlabel('Predicted PCA Loadings', fontsize=10)
        ax[0].set_ylabel('Actual PCA Loadings', fontsize=10)

        ax[1].set_xlabel('Predicted FCCA Loadings', fontsize=10)
        ax[1].set_ylabel('Actual FCCA Loadings', fontsize=10)

        ax[0].annotate(r'$\beta$ '  + '(var, dec, enc): (%f, %f, %f)' % tuple(coefp), (0.05, 0.75), xycoords='axes fraction', fontsize=7)
        ax[1].annotate(r'$\beta$ '  + '(var, dec, enc): (%f, %f, %f)' % tuple(coeff), (0.05, 0.75), xycoords='axes fraction', fontsize=7)

        fig.tight_layout()
        fig_save_path = '%s/%i_noact_%s.pdf' % (figpath, i, region)
        fig.savefig(fig_save_path, bbox_inches='tight', pad_inches=0)
    
def make_plot_2(coefp, coeff, figpath, region):

        # Make a boxplot of the coefficients
        fig, ax = plt.subplots(1, 2, figsize=(5, 5))

        ax[0].boxplot(np.array(coefp), showfliers=False)
        ax[1].boxplot(np.array(coeff), showfliers=False)
        ax[0].set_title('PCA Pred. Coef', fontsize=8)
        ax[1].set_title('FCCA Pred. Coef', fontsize=8)
        ax[0].set_xticklabels(['var', 'dec', 'enc'])
        ax[1].set_xticklabels(['var', 'dec', 'enc'])

        fig_save_path = '%s/coef_boxplot_noact_%s.pdf' % (figpath, region)
        fig.savefig(fig_save_path, bbox_inches='tight', pad_inches=0)

def make_plot_3(r1f_, r1p_, figpath, region):

    # Make a boxplot out of it
    fig, ax = plt.subplots(1, 1, figsize=(1, 5))
    medianprops = dict(linewidth=1, color='b')
    whiskerprops = dict(linewidth=0)
    bplot = ax.boxplot([r1f_, r1p_], patch_artist=True, medianprops=medianprops, notch=False, showfliers=False, whiskerprops=whiskerprops, showcaps=False)
    ax.set_xticklabels(['FBC', 'FFC'], rotation=45)
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylabel('Spearman ' + r'$\rho$', fontsize=18)

    colors = ['r', 'k']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    fig_save_path = '%s/iscore_prediction_boxplot_noact_%s.pdf' % (figpath, region)
    fig.savefig(fig_save_path, bbox_inches='tight', pad_inches=0)



def make_plot_4(su_r, DIM, figpath, region):

    # Prior to averaging, run tests. Updated for multiple comparisons adjustment. 
    _, p1 = scipy.stats.wilcoxon(su_r[:, 0, 0], su_r[:, 1, 0], alternative='less')
    _, p2 = scipy.stats.wilcoxon(su_r[:, 0, 1], su_r[:, 1, 1], alternative='less')
    _, p3 = scipy.stats.wilcoxon(su_r[:, 0, 2], su_r[:, 1, 2], alternative='less')
    _, p4 = scipy.stats.wilcoxon(su_r[:, 0, 3], su_r[:, 1, 3], alternative='less')
    pvec = np.sort([p1, p3, p4])  
    a1 = pvec[0] * 3
    a2 = pvec[1] * 2
    a3 = pvec[2]
    std_err = np.std(su_r, axis=0).ravel()/np.sqrt(35)
    su_r = np.mean(su_r, axis=0).ravel()

    # Permute so that each statistic is next to each other. No ACT.
    su_r = su_r[[0, 4, 2, 6, 3, 7]]
    std_err = std_err[[0, 4, 2, 6, 3, 7]]

    fig, ax = plt.subplots(figsize=(3, 5),)
    bars = ax.bar([0, 1, 3, 4, 6, 7],
                    su_r,
                    color=['r', 'k', 'r', 'k', 'r', 'k'], alpha=0.65,
                    yerr=std_err, capsize=5)

    # Place numerical values above the bars
    ax.text(0.5, 1.0, '****', ha='center')
    ax.text(3.5, 0.6, '****', ha='center')
    ax.text(6.5, 0.76, '****', ha='center')
    ax.set_ylim([-0.5, 1.1])
    ax.set_xticks([0.5, 3.5, 6.5])
    ax.set_xticklabels(['S.U. Var.', 'Dec. Weights', 'S.U. Enc. ' + r'$r^2$'], rotation=30, fontsize=12, ha='right')

    # Manual creation of legend
    colors = ['r', 'k']
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.65) for c in colors]
    labels = ['FBC', 'FFC']
    ax.legend(handles, labels, loc='lower right', prop={'size': 14})
    ax.set_ylabel('Spearman Correlation ' + r'$\rho$', fontsize=18)
    ax.set_yticks([-0.5, 0, 0.5, 1.])
    ax.tick_params(axis='both', labelsize=16)

    # Horizontal line at 0
    ax.hlines(0, -0.5, 7.5, color='k')
    fig_save_path = '%s/su_spearman_d%d_%s.pdf' % (figpath, DIM, region)
    fig.savefig(fig_save_path, bbox_inches='tight', pad_inches=0)


def make_plot_5(ss_angles, dims, fcca_delta_marg, pca_delta_marg, DIM, figpath):

    fig = plt.figure(figsize=(7, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax = [ax0, ax1]


    ############################ Ax 0: Subspace angles over Marginals ############################
    medianprops = dict(linewidth=1, color='b')
    whiskerprops = dict(linewidth=0)
    bplot = ax[0].boxplot([np.mean(ss_angles[:, :, 2, :], axis=-1).ravel(), np.mean(ss_angles[:, :, 1, :], axis=-1).ravel()], 
                    patch_artist=True, medianprops=medianprops, notch=False, showfliers=False,
                    whiskerprops=whiskerprops, showcaps=False)
    ax[0].set_xticklabels(['FBC/FBCm', 'FFC/FFCm'], rotation=30)
    for label in ax[0].get_xticklabels():
        label.set_horizontalalignment('center')
    ax[0].set_ylim([0, np.pi/2])
    ax[0].set_yticks([0, np.pi/8, np.pi/4, 3 * np.pi/8, np.pi/2])
    ax[0].set_yticklabels(['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    ax[0].tick_params(axis='both', labelsize=16)
    ax[0].set_ylabel('Subspace angles (rads)', fontsize=18)
    colors = ['r', 'k']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)


    ############################ Ax 1: Decoding over Marginals ############################
    sqrt_norm_val = 28
    colors = ['black', 'red', '#781820', '#5563fa']

    # FCCA and PCADecoding Over Marginals
    ax[1].plot(dims, np.mean(fcca_delta_marg, axis=0), color=colors[1])
    ax[1].plot(dims, np.mean(pca_delta_marg, axis=0), color=colors[0])


    ax[1].fill_between(dims, np.mean(fcca_delta_marg, axis=0) + np.std(fcca_delta_marg, axis=0)/np.sqrt(sqrt_norm_val),
                    np.mean(fcca_delta_marg, axis=0) - np.std(fcca_delta_marg, axis=0)/np.sqrt(sqrt_norm_val), color=colors[1], alpha=0.25, label='__nolegend__')
    ax[1].fill_between(dims, np.mean(pca_delta_marg, axis=0) + np.std(pca_delta_marg, axis=0)/np.sqrt(sqrt_norm_val),
                    np.mean(pca_delta_marg, axis=0) - np.std(pca_delta_marg, axis=0)/np.sqrt(sqrt_norm_val), color=colors[0], alpha=0.25, label='__nolegend__')



    ax[1].set_xlabel('Dimension', fontsize=18)
    ax[1].set_ylabel(r'$\Delta$' + ' Stim ID Decoding ', fontsize=18, labelpad=-2)
    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    #ax[1].set_ylim([0, 0.2])
    #ax[1].set_yticks([0, 0.2])
    ax[1].legend(['FBC/FBCm', 'FFC/FFCm'], loc='lower right', fontsize=14)


    # Some Statistical Tests
    comp_dim_ind = np.argwhere(dims == DIM)[0][0]
    stat, p = scipy.stats.wilcoxon(fcca_delta_marg[:, comp_dim_ind], pca_delta_marg[:, comp_dim_ind], alternative='greater')
    #print('Delta decoding p=%f' % p)
    #print(np.mean(fcca_delta_marg[:, comp_dim_ind]) - np.mean(pca_delta_marg[:, comp_dim_ind]))

    fig.tight_layout()
    fig_save_path = '%s/decoding_differences_%s.pdf' % (figpath, region)
    fig.savefig(fig_save_path, bbox_inches='tight', pad_inches=0)
    


def make_Fig5_bcd(decode_df, data_path, region, DIM, figpath='.', debug_plots=False):

    # Get Data 
    bin_width = decode_df['loader_args'][0]['bin_width']
    _, loadings_df = get_top_neurons(decode_df, method1='FCCA', method2='PCA', T=3, n=5, manual_dim=DIM, pairwise_exclude=False, data_path=data_path, bin_width=bin_width, region=region)
    su_cals_df = get_su_calcs(region)

    itrim_df = loadings_df
    data_files = np.unique(itrim_df['data_file'].values)
    stats = ['su_var', 'su_act', 'decoding_weights', 'su_encoding_r2']

    ############################ Start Filling in cArray
    carray = []
    for i, data_file in enumerate(data_files):
        df = apply_df_filters(itrim_df, data_file=data_file)
        carray_ = np.zeros((df.shape[0], len(stats)))
        for j in range(df.shape[0]):                    # Find the corFrelaton between 
            for k, stat in enumerate(stats):
                
                # Grab the unique identifiers needed
                nidx = df.iloc[j]['nidx']
                df_ = apply_df_filters(su_cals_df, data_file=data_file)
                carray_[j, k] = get_scalar(df_, stat, nidx)
        carray.append(carray_)


    ############################ Start Filling in su_r
    su_r = np.zeros((len(carray), 2, carray[0].shape[1]))
    keys = ['FCCA_loadings', 'PCA_loadings']
    X, Yf, Yp, x_, yf_, yp_ = [], [], [], [], [], []
    for i in range(len(carray)):
        for j in range(2):
            df = apply_df_filters(itrim_df, data_file=data_files[i])
            x1 = df[keys[j]].values

            if j == 0:
                Yf.extend(x1)
                yf_.append(x1)
            else:
                Yp.extend(x1)
                yp_.append(x1)

            xx = []

            for k in range(carray[0].shape[1]):
                x2 = carray[i][:, k]
                xx.append(x2)
                su_r[i, j, k] = scipy.stats.spearmanr(x1, x2)[0]
        
            xx = np.array(xx).T            
        X.append(xx)
        x_.append(xx)

    X = np.vstack(X)
    Yf = np.array(Yf)[:, np.newaxis]
    Yp = np.array(Yp)[:, np.newaxis]
    assert(X.shape[0] == Yf.shape[0])
    assert(X.shape[0] == Yp.shape[0])

    ############################ Train a linear model to predict loadings from the single unit statistics and then assess the spearman correlation between predicted and actual loadings
    r1p_, r1f_, coefp, coeff, rpcv, rfcv = [], [], [], [], [], []

    for i in range(len(carray)):

        linmodel = LinearRegression().fit(x_[i][:, [0, 2, 3]], np.array(yp_[i])[:, np.newaxis])
        linmodel2 = LinearRegression().fit(x_[i][:, [0, 2, 3]], np.array(yf_[i])[:, np.newaxis])

        yp_pred = linmodel.predict(x_[i][:, [0, 2, 3]])
        yf_pred = linmodel2.predict(x_[i][:, [0, 2, 3]])

        # get normalized coefficients for feature importance assessment
        x__ = StandardScaler().fit_transform(x_[i][:, [0, 2, 3]])
        y__ = StandardScaler().fit_transform(np.array(yp_[i])[:, np.newaxis])

        linmodel = LinearRegression().fit(x__, y__)
        coefp.append(linmodel.coef_.squeeze())

        y__ = StandardScaler().fit_transform(np.array(yf_[i])[:, np.newaxis])
        linmodel = LinearRegression().fit(x__, y__)
        coeff.append(linmodel.coef_.squeeze())

        # Try cross-validation
        rpcv.append(np.mean(cross_val_score(LinearRegression(), x_[i][:, [0, 2, 3]], np.array(yp_[i])[:, np.newaxis], cv=5)))
        rfcv.append(np.mean(cross_val_score(LinearRegression(), x_[i][:, [0, 2, 3]], np.array(yf_[i])[:, np.newaxis], cv=5)))
        r1p_.append(scipy.stats.spearmanr(yp_pred.squeeze(), np.array(yp_[i]).squeeze())[0])
        r1f_.append(scipy.stats.spearmanr(yf_pred.squeeze(), np.array(yf_[i]).squeeze())[0])


        if debug_plots: 
            make_plot_1(yp_pred, yf_pred, yp_[i], yf_[i], r1p_[i], r1f_[i], coefp[i], coeff[i], i, figpath, region)


    ############################ Run Stats
    stats, p = scipy.stats.wilcoxon(r1p_, r1f_, alternative='greater')

    if debug_plots: 
        make_plot_2(coefp, coeff, figpath, region)

    # Make Histogram
    make_plot_3(r1f_, r1p_, figpath, region)

    # Get predictions:
    linmodel1 = LinearRegression().fit(X, Yp)
    linmodel2 = LinearRegression().fit(X, np.log10(Yp))

    Yp_pred1 = linmodel1.predict(X)
    Yp_pred2 = linmodel2.predict(X)

    r1p = scipy.stats.spearmanr(Yp_pred1.squeeze(), Yp.squeeze())[0]
    r2p = scipy.stats.spearmanr(Yp_pred2.squeeze(), Yp.squeeze())[0]

    linmodel1 = LinearRegression().fit(X, Yf)
    linmodel2 = LinearRegression().fit(X, np.log10(Yf))

    Yf_pred1 = linmodel1.predict(X)
    Yf_pred2 = linmodel2.predict(X)

    r1f = scipy.stats.spearmanr(Yf_pred1.squeeze(), Yf.squeeze())[0]
    r2f = scipy.stats.spearmanr(Yf_pred2.squeeze(), Yf.squeeze())[0]

    make_plot_4(su_r, DIM, figpath, region)


def make_Fig5_egfh(decoding_df, data_path, region, DIM, figpath='.', debug_plots=True):

    marginal_df = get_marginal_dfs(region)

    if region in ['AM', 'ML']:
        metric = 'loss' # this is the df key for what is being decoded
    

    ####################### Find average decoding differences over marginals
    fold_idcs = np.unique(decoding_df['fold_idx'].values)
    data_files = np.unique(decoding_df['data_file'].values)
    dims = np.unique(decoding_df['dim'].values)
    methods = np.unique(decoding_df['dimreduc_method'].values)

    decoding_structs_FFC = np.zeros((data_files.size, fold_idcs.size, dims.size))
    decoding_structs_FBC = np.zeros((data_files.size, fold_idcs.size, dims.size))

    decoding_structs_marginal_FFC = np.zeros((data_files.size, fold_idcs.size, dims.size))
    decoding_structs_marginal_FBC = np.zeros((data_files.size, fold_idcs.size, dims.size))


    for df_ind, data_file in tqdm(enumerate(data_files)):
        for fold_ind, fold in enumerate(fold_idcs):
            for dim_ind, dim in enumerate(dims):
            
                decoding_structs_FFC[df_ind, fold_ind, dim_ind] = 1 - apply_df_filters(decoding_df, data_file=data_file, dim=dim, dimreduc_method='PCA', loader_args={'region': region},  fold_idx=fold)[metric].iloc[0]
                decoding_structs_FBC[df_ind, fold_ind, dim_ind] = 1 - apply_df_filters(decoding_df, data_file=data_file, dim=dim, dimreduc_method='LQGCA', loader_args={'region': region},  fold_idx=fold)[metric].iloc[0]
                
                decoding_structs_marginal_FFC[df_ind, fold_ind, dim_ind] = 1 - apply_df_filters(marginal_df, data_file=data_file, dim=dim, dimreduc_method='PCA', loader_args={'region': region},  fold_idx=fold)[metric].iloc[0]
                decoding_structs_marginal_FBC[df_ind, fold_ind, dim_ind] = 1 - apply_df_filters(marginal_df, data_file=data_file, dim=dim, dimreduc_method='LQGCA', loader_args={'region': region},  fold_idx=fold)[metric].iloc[0]


    apply_df_filters(marginal_df, data_file=data_file, dim=dim, dimreduc_method='PCA', loader_args={'region': region},  fold_idx=fold)
    apply_df_filters(marginal_df, data_file=data_file, fold_idx=fold)

    ####################### Average across folds and get deltas
    pca_dec = np.mean(decoding_structs_FFC, axis=1).squeeze()
    fcca_dec = np.mean(decoding_structs_FBC, axis=1).squeeze()

    pca_marginal_dec = np.mean(decoding_structs_marginal_FFC, axis=1).squeeze()
    fcca_marginal_dec = np.mean(decoding_structs_marginal_FBC, axis=1).squeeze()

    pca_dec = pca_dec.reshape(data_files.size, -1)
    fcca_dec = fcca_dec.reshape(data_files.size, -1)
    pca_marginal_dec = pca_marginal_dec.reshape(data_files.size, -1)
    fcca_marginal_dec = fcca_marginal_dec.reshape(data_files.size, -1)

    fcca_delta_marg = fcca_dec - fcca_marginal_dec
    pca_delta_marg = pca_dec - pca_marginal_dec 


    ####################### Find average subspace angles over marginals
    ss_angles = np.zeros((data_files.size, fold_idcs.size, 4, DIM))

    for df_ind, data_file in tqdm(enumerate(data_files)):
        for fold_ind, fold in enumerate(fold_idcs):

            dfpca = apply_df_filters(decoding_df, data_file=data_file, dimreduc_method='PCA', loader_args={'region': region}, fold_idx=fold, dim=DIM)
            dffcca = apply_df_filters(decoding_df, data_file=data_file, dimreduc_method='LQGCA', loader_args={'region': region}, fold_idx=fold, dim=DIM)

            dfpca_marginal = apply_df_filters(marginal_df, data_file=data_file, dimreduc_method='PCA', fold_idx=fold, loader_args={'region': region}, dim=DIM)
            dffca_marginal = apply_df_filters(marginal_df, data_file=data_file, dimreduc_method='LQGCA', fold_idx=fold, loader_args={'region': region}, dim=DIM)

            ss_angles[df_ind, fold_ind, 0, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:DIM], dffcca.iloc[0]['coef'])
            ss_angles[df_ind, fold_ind, 1, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:DIM], dfpca_marginal.iloc[0]['coef'][:, 0:DIM])
            ss_angles[df_ind, fold_ind, 2, :] = scipy.linalg.subspace_angles(dffcca.iloc[0]['coef'], dffca_marginal.iloc[0]['coef'])
            ss_angles[df_ind, fold_ind, 3, :] = scipy.linalg.subspace_angles(dffca_marginal.iloc[0]['coef'], dfpca_marginal.iloc[0]['coef'][:, 0:DIM])

    stat, p1 = scipy.stats.wilcoxon(np.mean(ss_angles[:, :, 2, :], axis=-1).ravel(), np.mean(ss_angles[:, :, 1, :], axis=-1).ravel(), alternative='greater')
    stat, p2 = scipy.stats.wilcoxon(np.mean(ss_angles[:, :, 2, :], axis=-1).ravel(), np.mean(ss_angles[:, :, 0, :], axis=-1).ravel(), alternative='greater')
    #print(p1,p2)

    make_plot_5(ss_angles, dims, fcca_delta_marg, pca_delta_marg, DIM, figpath)


if __name__ == '__main__':
    #region = 'HPC'
    #figpath = '/home/akumar/nse/neural_control/figs/revisions'

    region = 'AM'
    DIM = 21
    figpath = '/home/marcush/projects/neural_control/notebooks/TsaoLabNotebooks/CodeAndFigsForPaper/Figs'


    df, session_key = load_decoding_df(region)
    data_path = get_data_path(region)

    make_Fig5_bcd(df, data_path, region, DIM, figpath=figpath, debug_plots=False)
    make_Fig5_egfh(df, data_path, region, DIM, figpath=figpath, debug_plots=False)

