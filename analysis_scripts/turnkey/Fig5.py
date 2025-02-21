import pdb
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy 
import pickle
import pandas as pd
from statsmodels.stats import multitest


from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from config import PATH_DICT
sys.path.append(PATH_DICT['repo'])
from utils import calc_loadings
from region_select import *
from Fig4 import get_loadings_df
from decodingvdim import get_decoding_performance

def get_su_calcs(region):
    if region in ['M1_psid', 'S1_psid']:
        region = region.split('_psid')[0]

    su_calcs_path = PATH_DICT['tmp'] + '/su_calcs_%s.pkl' % region
    with open(su_calcs_path, 'rb') as f:
        su_stats = pickle.load(f)

    su_calcs_df = pd.DataFrame(su_stats)
    return su_calcs_df

def get_marginal_dfs(region):
    # Fill in directories for marginals:
    root_path = PATH_DICT['df']
    if region == 'M1_psid':
        # with open(root_path + '/indy_decoding_marginal.dat', 'rb') as f:
        #     rl = pickle.load(f)
        # indy_mdf = pd.DataFrame(rl)

        # with open(root_path + '/loco_decoding_marginal_df.dat', 'rb') as f:
        #     rl = pickle.load(f)
        # loco_mdf = pd.DataFrame(rl)
        # marginals_df = pd.concat([indy_mdf, loco_mdf])

        with open(root_path + '/sabes_marginal_psid_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        # Filter by start time truncation only
        filt = [idx for idx in range(df.shape[0]) 
                if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        df = df.iloc[filt]
        df_pca = apply_df_filters(df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(df, dimreduc_args={'T':3, 'loss_type':'trace', 'n_init':10})
        marginals_df = pd.concat([df_pca, df_fcca])
    elif region == 'S1_psid':
        with open(root_path + '/sabes_marginal_psid_dfS1.pkl', 'rb') as f:
            rl = pickle.load(f)
        df = pd.DataFrame(rl)
        # Filter by start time truncation only
        filt = [idx for idx in range(df.shape[0]) 
                if df.iloc[idx]['loader_args']['subset'] is None and df.iloc[idx]['loader_args']['truncate_start'] is True]
        marginals_df = df.iloc[filt]
    elif region == 'M1_trialized':
        raise NotImplementedError
    elif region == 'HPC_peanut':
        with open(root_path + '/peanut_marginal_decoding25_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        marginals_df = pd.DataFrame(rl)


        df_pca = apply_df_filters(marginals_df, dimreduc_method='PCA')
        df_fcca = apply_df_filters(marginals_df, dimreduc_args={'T':5, 
              'loss_type':'trace', 'n_init':10, 'marginal_only':True})
        marginals_df = pd.concat([df_pca, df_fcca])

        filt = [idx for idx in range(marginals_df.shape[0])
                if marginals_df.iloc[idx]['decoder_args']['decoding_window'] == 12]
        marginals_df = marginals_df.iloc[filt]


        # Unpack the epoch from the loader_args
        epochs = [marginals_df.iloc[k]['loader_args']['epoch'] for k in range(marginals_df.shape[0])]
        marginals_df['epoch'] = epochs

    elif region == 'M1_maze':
        with open(root_path + '/mcmaze_marginal_decoding_df.pkl', 'rb') as f:
            rl = pickle.load(f)
        marginals_df = pd.DataFrame(rl)
        
        # Hard-coded match to the non-marginal mc maze arguments
        loader_args = {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':1,
        'trialize':True, 'interval':'after_go', 'trial_threshold':0.5} 
        decoder_args = {'trainlag': 3, 'testlag': 3, 'decoding_window':5}
        marginals_df = apply_df_filters(marginals_df, loader_args=loader_args, decoder_args = decoder_args)
    elif region in ['AM', 'ML']:
        marginals_path = root_path + '/tsao_marginal_decode_df.pkl'
        #marginals_path = root_path + '/tsao_decode_df_clearOnly_marginals.pkl' #'/tsao_decode_df_degOnly_marginals.pkl'
        
        
        with open(marginals_path, 'rb') as f:
            rl = pickle.load(f)
        marginals_df = pd.DataFrame(rl)
        marginals_df = apply_df_filters(marginals_df, loader_args={'region': region})
        
    elif region in ['HPC', 'mPFC']:
        
        marginals_path = root_path + '/decoding_fullarg_frank_lab_marginals_glom.pickle'
        
        with open(marginals_path, 'rb') as f:
            rl = pickle.load(f)
        marginals_df = pd.DataFrame(rl)
        
        load_idx, dec_idx, dr_idx = loader_kwargs[region].values()
        loader_args, decoder_args, dimreduc_args = get_franklab_args(load_idx, dec_idx, dr_idx)
        
        marginals_df_PCA = apply_df_filters(marginals_df, loader_args=loader_args, decoder_args=decoder_args, dimreduc_method="PCA")
        marginals_df_LQGCA = apply_df_filters(marginals_df, loader_args=loader_args, decoder_args=decoder_args, dimreduc_args=dimreduc_args)        
        marginals_df = pd.concat([marginals_df_PCA, marginals_df_LQGCA])       
    
    elif region in ['VISp']:
        
        marginals_path = root_path + '/decoding_AllenVC_VISp_marginals_glom.pickle'        
        
        with open(marginals_path, 'rb') as f:
            rl = pickle.load(f)
        marginals_df = pd.DataFrame(rl)
        
        unique_loader_args = list({frozenset(d.items()) for d in marginals_df['loader_args']})
        loader_args=dict(unique_loader_args[loader_kwargs[region]['load_idx']])
        marginals_df = apply_df_filters(marginals_df, **{'loader_args':loader_args})
                
        
    return marginals_df    

def get_scalar(df_, stat, neu_idx):
    neu_idx = int(neu_idx)
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
    if region == 'ML':
        ax.set_ylim([-0.25, 1])
    else:
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
    stats = ['Var', 'Dec', 'Enc']
    porder = np.argsort([p1, p3, p4])

    a1 = pvec[0] * 3
    a2 = pvec[1] * 2
    a3 = pvec[2]

    print('Histogram stats:\n')
    print(', '.join([f'{stats[porder[0]]}:{a1}',
                     f'{stats[porder[1]]}:{a2}',
                     f'{stats[porder[2]]}:{a3}']))

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
    # ax.text(0.5, 1.0, '****', ha='center')
    # ax.text(3.5, 0.6, '****', ha='center')
    # ax.text(6.5, 0.76, '****', ha='center')
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
    # ax[1].set_ylabel(r'$\Delta$' + ' Stim ID Decoding ', fontsize=18, labelpad=-10)
    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    
    # Start Dimension at 1
    ax[1].set_xlim([1, 30])
    ax[1].set_xticks([1, 10, 20, 30])

    #ax[1].set_ylim([0, 0.2])
    #ax[1].set_yticks([0, 0.2])
    ax[1].legend(['FBC/FBCm', 'FFC/FFCm'], loc='lower right', fontsize=14)

    # Some Statistical Tests
    comp_dim_ind = np.argwhere(dims == DIM)[0][0]
    pkr2f = fcca_delta_marg[:, comp_dim_ind]
    pkr2p = pca_delta_marg[:, comp_dim_ind]

    stat, p = scipy.stats.wilcoxon(fcca_delta_marg[:, comp_dim_ind], pca_delta_marg[:, comp_dim_ind], alternative='greater')
    print('Delta decoding p=%f' % p)
    #print(np.mean(fcca_delta_marg[:, comp_dim_ind]) - np.mean(pca_delta_marg[:, comp_dim_ind]))

    fig.tight_layout()
    fig_save_path = '%s/decoding_differences_%s.pdf' % (figpath, region)
    fig.savefig(fig_save_path, bbox_inches='tight', pad_inches=0)
    return pkr2f, pkr2p

def make_Fig5_bcd(decode_df_all, session_key, data_path, region, DIM, figpath='.', debug_plots=False):

    # Load dimreduc_df and calculate loadings
    unique_loader_args = list({frozenset(d.items()) for d in decode_df_all['loader_args']})
    loader_args=dict(unique_loader_args[loader_kwargs[region]['load_idx']])
    decode_df = apply_df_filters(decode_df_all, **{'loader_args':loader_args})
    
    
    loadings_df = get_loadings_df(decode_df, session_key, DIM)
    su_calcs_df = get_su_calcs(region)
    sessions = np.unique(loadings_df[session_key].values)
    if region in ['AM', 'ML', 'VISp']:
        stats = ['su_var', 'su_act', 'decoding_weights', 'su_encoding_r2']
    else:
        stats = ['su_var', 'su_act', 'decoding_weights', 'su_r2_enc']
    ############################ Start Filling in cArray
    carray = []
    for i, session in enumerate(sessions):
            
    
        df_filter = {session_key:sessions[i]}
        df = apply_df_filters(loadings_df, **df_filter)
        carray_ = np.zeros((df.shape[0], len(stats)))
        for j in range(df.shape[0]):                    # Find the corFrelaton between 
            for k, stat in enumerate(stats):
                # Grab the unique identifiers needed
                nidx = df.iloc[j]['nidx']
                try:
                    df_ = apply_df_filters(su_calcs_df, **{session_key:session})
                except:
                    df_ = apply_df_filters(su_calcs_df, session=session)
                carray_[j, k] = get_scalar(df_, stat, nidx)
        carray.append(carray_)


    ############################ Start Filling in su_r
    su_r = np.zeros((len(carray), 2, carray[0].shape[1]))
    keys = ['FCCA_loadings', 'PCA_loadings']
    X, Yf, Yp, x_, yf_, yp_ = [], [], [], [], [], []
    for i in range(len(carray)):
        for j in range(2):
            df_filter = {session_key:sessions[i]}
            df = apply_df_filters(loadings_df, **df_filter)
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
    print(f'S.U. prediction medians:({np.median(r1p_)}, {np.median(r1f_)})')
    print(f'S.U. prediction test:{p}')

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


def make_Fig5_egfh(decoding_df, session_key, data_path, region, DIM, figpath='.', debug_plots=True):

    marginal_df = get_marginal_dfs(region)

    ####################### Find average decoding differences over marginals
    fold_idcs = np.unique(decoding_df['fold_idx'].values)
    sessions = np.unique(decoding_df[session_key].values)
    dims = np.unique(decoding_df['dim'].values)

    decoding_structs_FFC = np.zeros((sessions.size, fold_idcs.size, dims.size))
    decoding_structs_FBC = np.zeros((sessions.size, fold_idcs.size, dims.size))

    decoding_structs_marginal_FFC = np.zeros((sessions.size, fold_idcs.size, dims.size))
    decoding_structs_marginal_FBC = np.zeros((sessions.size, fold_idcs.size, dims.size))

    for df_ind, session in tqdm(enumerate(sessions)):
        for fold_ind, fold in enumerate(fold_idcs):
            for dim_ind, dim in enumerate(dims):
                df_filter = {session_key:session, 'dim':dim, 'dimreduc_method':'PCA', 'fold_idx':fold}
                df_ = apply_df_filters(decoding_df, **df_filter)
                assert(df_.shape[0] == 1)
                decoding_structs_FFC[df_ind, fold_ind, dim_ind] = get_decoding_performance(df_, region)

                df_filter = {session_key:session, 'dim':dim, 'dimreduc_method':['LQGCA', 'FCCA'], 'fold_idx':fold}
                df_ = apply_df_filters(decoding_df, **df_filter)
                assert(df_.shape[0] == 1)
                decoding_structs_FBC[df_ind, fold_ind, dim_ind] = get_decoding_performance(df_, region)
                
                df_filter = {session_key:session, 'dim':dim, 'dimreduc_method':'PCA', 'fold_idx':fold}
                df_ = apply_df_filters(marginal_df, **df_filter)
                assert(df_.shape[0] == 1)
                decoding_structs_marginal_FFC[df_ind, fold_ind, dim_ind] = get_decoding_performance(df_, region)

                df_filter = {session_key:session, 'dim':dim, 'dimreduc_method':['LQGCA', 'FCCA'], 'fold_idx':fold}
                df_ = apply_df_filters(marginal_df, **df_filter)
                assert(df_.shape[0] == 1)
                decoding_structs_marginal_FBC[df_ind, fold_ind, dim_ind] = get_decoding_performance(df_, region)

    ####################### Average across folds and get deltas
    pca_dec = np.mean(decoding_structs_FFC, axis=1).squeeze()
    fcca_dec = np.mean(decoding_structs_FBC, axis=1).squeeze()

    pca_marginal_dec = np.mean(decoding_structs_marginal_FFC, axis=1).squeeze()
    fcca_marginal_dec = np.mean(decoding_structs_marginal_FBC, axis=1).squeeze()

    pca_dec = pca_dec.reshape(sessions.size, -1)
    fcca_dec = fcca_dec.reshape(sessions.size, -1)
    pca_marginal_dec = pca_marginal_dec.reshape(sessions.size, -1)
    fcca_marginal_dec = fcca_marginal_dec.reshape(sessions.size, -1)

    fcca_delta_marg = fcca_dec - fcca_marginal_dec
    pca_delta_marg = pca_dec - pca_marginal_dec 

    ####################### Find average subspace angles over marginals
    ss_angles = np.zeros((sessions.size, fold_idcs.size, 4, DIM))

    for df_ind, session in tqdm(enumerate(sessions)):
        for fold_ind, fold in enumerate(fold_idcs):
            df_filter = {session_key:session, 'dim':DIM, 'dimreduc_method':'PCA', 'fold_idx':fold}
            dfpca = apply_df_filters(decoding_df, **df_filter)
            assert(dfpca.shape[0] == 1)

            df_filter = {session_key:session, 'dim':DIM, 'dimreduc_method':['LQGCA', 'FCCA'], 'fold_idx':fold}
            dffcca = apply_df_filters(decoding_df, **df_filter)
            assert(dffcca.shape[0] == 1)

            df_filter = {session_key:session, 'dim':DIM, 'dimreduc_method':'PCA', 'fold_idx':fold}
            dfpca_marginal = apply_df_filters(marginal_df, **df_filter)
            try:
                assert(dfpca_marginal.shape[0] == 1)
            except:
                pdb.set_trace()

            df_filter = {session_key:session, 'dim':DIM, 'dimreduc_method':['LQGCA', 'FCCA'], 'fold_idx':fold}
            dffcca_marginal = apply_df_filters(marginal_df, **df_filter)
            try:
                assert(dffcca_marginal.shape[0] == 1)
            except:
                pdb.set_trace()
            ss_angles[df_ind, fold_ind, 0, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:DIM], dffcca.iloc[0]['coef'])
            ss_angles[df_ind, fold_ind, 1, :] = scipy.linalg.subspace_angles(dfpca.iloc[0]['coef'][:, 0:DIM], dfpca_marginal.iloc[0]['coef'][:, 0:DIM])
            ss_angles[df_ind, fold_ind, 2, :] = scipy.linalg.subspace_angles(dffcca.iloc[0]['coef'], dffcca_marginal.iloc[0]['coef'])
            ss_angles[df_ind, fold_ind, 3, :] = scipy.linalg.subspace_angles(dffcca_marginal.iloc[0]['coef'], dfpca_marginal.iloc[0]['coef'][:, 0:DIM])

    stat, p1 = scipy.stats.wilcoxon(np.mean(ss_angles[:, :, 2, :], axis=-1).ravel(), np.mean(ss_angles[:, :, 1, :], axis=-1).ravel(), alternative='greater')
    stat, p2 = scipy.stats.wilcoxon(np.mean(ss_angles[:, :, 2, :], axis=-1).ravel(), np.mean(ss_angles[:, :, 0, :], axis=-1).ravel(), alternative='greater')

    print(f'marginal ssa p vals: FBC vs. FBCm/FFC vs. FFCm: {p1}')
    pkr2f, pkr2p = make_plot_5(ss_angles, dims, fcca_delta_marg, pca_delta_marg, DIM, figpath)
    return pkr2f, pkr2p

dim_dict = {
    'M1_psid': 6,
    'M1_trialized':6,
    'S1_psid': 6,
    'HPC_peanut': 11,
    'M1_maze': 6,
    'AM': 21,
    'ML': 21,
    'mPFC':15,
    'HPC':15,
    'VISp':10
}

if __name__ == '__main__':
    # regions = ['M1', 'S1', 'M1_maze', 'HPC_peanut', 'AM', 'ML']
    regions = ['VISp']
    for region in tqdm(regions):
        DIM = dim_dict[region]
        figpath = PATH_DICT['figs']

        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        data_path = get_data_path(region)
        make_Fig5_bcd(df, session_key, data_path, region, DIM, figpath=figpath, debug_plots=False)
        pkr2f, pkr2p = make_Fig5_egfh(df, session_key, data_path, region, DIM, figpath=figpath, debug_plots=False)

        deltar2f_all.append(pkr2f)
        deltar2p_all.append(pkr2p)

    # Test that M1 is stochastically greater than both S1 and HPC for FBC...
    _, p1 = scipy.stats.mannwhitneyu(deltar2f_all[0], deltar2f_all[1], alternative='greater')
    _, p2 = scipy.stats.mannwhitneyu(deltar2f_all[0], deltar2f_all[2], alternative='greater')

    # ..and FFC
    _, p3 = scipy.stats.mannwhitneyu(deltar2p_all[0], deltar2p_all[1], alternative='greater')
    _, p4 = scipy.stats.mannwhitneyu(deltar2p_all[0], deltar2p_all[2], alternative='greater')

    _, pcorrected1, _, _ = multitest.multipletests([p1, p2], method='holm')
    _, pcorrected2, _, _ = multitest.multipletests([p3, p4], method='holm')
    
    pdb.set_trace()

