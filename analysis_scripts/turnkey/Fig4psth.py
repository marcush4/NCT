import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import scipy 
import time
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import sys, os

from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#from npeet.entropy_estimators import entropy as knn_entropy
from dca.cov_util import calc_cross_cov_mats_from_data, calc_pi_from_cross_cov_mats

from region_select import *
from config import PATH_DICT
sys.path.append(PATH_DICT['repo'])

from utils import apply_df_filters, calc_loadings
from loaders import load_sabes
from segmentation import reach_segment_sabes

from Fig4 import get_loadings_df

ylim_dict = {
    'M1': [-0.52, 0.52],
    'S1': [-0.52, 0.52],
    'HPC_peanut': [-1.02, 1.02],
    'AM': [-0.52, 0.52],
    'ML': [-0.52, 0.52],
    'VISp': [-0.52, 0.52]
}

ytick_dict = {
    'M1': [-0.5, 0.5],
    'S1': [-0.5, 0.5],
    'HPC_peanut': [-1, 1],
    'AM': [-0.5, 0.5],
    'ML': [-0.5, 0.5],
    'VISp': [-0.5, 0.5]
}

xlim_dict = {
    'M1': [0, 1500],
    'S1': [0, 1500],
    'HPC_peanut': [0, 2500],
    'AM': [0, 500],
    'ML': [0, 500],
    'VISp': [0, 250]
}

n_dict = {
    'M1': 20,
    'S1': 20,
    'HPC_peanut': 30,
    'AM': 20,
    'ML': 20,
    'VISp': 20
}

xtick_dict = xlim_dict.copy()

def get_top_neurons(dimreduc_df, session_key, dim, fraction_cutoff=0.9, 
                    pairwise_exclude=True, n=10):

    sessions = np.unique(dimreduc_df[session_key].values)
    loadings_df = get_loadings_df(dimreduc_df, session_key, dim=dim)

    # For each data file, find the top 5 neurons that are high in one method but low in all others
    top_neurons_l = []
    for i, session in tqdm(enumerate(sessions)):
        df_ = apply_df_filters(loadings_df, **{session_key:session})
        FCCA_ordering = np.argsort(df_['FCCA_loadings'].values)
        PCA_ordering = np.argsort(df_['PCA_loadings'].values)
        
        rank_diffs = np.zeros((PCA_ordering.size,))
        for j in range(df_.shape[0]):            
            rank_diffs[j] = list(FCCA_ordering).index(j) - list(PCA_ordering).index(j)

        # Find the top n neurons according to all pairwise high/low orderings

        # User selects which pairwise comparison is desired
        method_dict = {'PCA': PCA_ordering, 'FCCA':FCCA_ordering}

        top1 = []
        top2 = []

        idx = 0
        while not np.all([len(x) >= n for x in [top1, top2]]):
            if idx >= len(FCCA_ordering):
                break
            idx += 1
            # Take neurons from the top ordering of each method. Disregard neurons that 
            # show up in all methods

            top1_ = FCCA_ordering[-idx]
            top2_ = PCA_ordering[-idx]

            if pairwise_exclude:
                if top1_ != top2_:
                    if top1_ not in top2:
                        top1.append(top1_)
                    if top2_ not in top1:
                        top2.append(top2_)
                else:
                    continue
            else:
                top1.append(top1_)
                top2.append(top2_)

        sz = min(len(top1), len(top2))
        top_neurons = np.zeros((2, sz)).astype(int)

        top_neurons[0, :] = top1[0:sz]
        top_neurons[1, :] = top2[0:sz]

        top_neurons_l.append({session_key:session, 'top_neurons': top_neurons}) 

    top_neurons_df = pd.DataFrame(top_neurons_l)
    
    return top_neurons_df, loadings_df

def PSTH_plot(top_neurons_df, region, data_path, session_key, df):

    if not os.path.exists(PATH_DICT['figs'] + f'/{region}_psth'):
        os.makedirs(PATH_DICT['figs'] + f'/{region}_psth')

    sessions = np.unique(top_neurons_df[session_key].values)
    # sessions = ['indy_20160624_03.mat', 'indy_20160930_02.mat']
    for h, session in enumerate(sessions):
        df_ = apply_df_filters(top_neurons_df, **{session_key:session})
        fArgs = df['full_arg_tuple'] if region in ['ML', 'AM'] else None 
        
        if region in ['VISp']:
            load_idx = loader_kwargs[region]['load_idx']
            unique_loader_args = list({frozenset(d.items()) for d in df['loader_args']})
            loader_args=dict(unique_loader_args[load_idx])
        else:
            loader_args = None
        
               
        x, time = get_rates_smoothed(data_path, region, session, return_t=True, std=True, sigma=2, loader_args=loader_args, full_arg_tuple=fArgs)        

        
        # In the end, always only plot 10 neurons. However, for HPC we may have to 
        # skip over overly quiescent neurons
        n = n_dict[region]
        colors = ['r', 'k']
        for i in range(2):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            n_ = 0
            for j in range(n):
                if n_ >= 20:
                    break
                x_ = x[df_.iloc[0]['top_neurons'][i, j]]
                if np.max(x_) < 0.05:
                    continue
                else:
                    ax.plot(time, x_, colors[i], alpha=0.5)
                    n_ += 1
            ax.set_ylim(ylim_dict[region])
            ax.set_yticks(ytick_dict[region])
            ax.set_xlim(xlim_dict[region])
            ax.set_xticks(xtick_dict[region])

            ax.set_xticklabels([])
            #ax.set_yticklabels([])
            ax.set_ylabel('Z-scored Response', fontsize=14, rotation=90, labelpad=10)
            x_label_text = f"{round(xlim_dict[region][1] / 1000, 2)}\nTime (s)"
            ax.text(1.0, 0.6, x_label_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')


            #ax.spines['left'].set_position('center')
            ax.spines['bottom'].set_position('center')

            # Eliminate upper and right axes
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            # Show ticks in the left and lower axes only
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            fig.savefig(PATH_DICT['figs'] + f'/{region}_psth' + f'/{session}_{i}.pdf',
                        bbox_inches='tight', pad_inches=0)

dim_dict = {
    'M1': 6,
    'S1': 6,
    'M1_trialized':6,
    'HPC_peanut': 11,
    'M1_maze':6,
    'AM': 21,
    'ML': 21,
    'mPFC': 5,
    'VISp':10
}

if __name__ == '__main__':

    figpath = PATH_DICT['figs']
    regions = ['AM', 'ML']
    for region in regions:    
        data_path = get_data_path(region)
        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        top_neurons_df, _ = get_top_neurons(df, session_key, dim_dict[region], 
                                           fraction_cutoff=0.5, n=n_dict[region])
        PSTH_plot(top_neurons_df, region, data_path, session_key, df)