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

from region_select import *
from config import PATH_DICT

rate_loader_kwargs = {
    'M1': dict(boxcox=None, high_pass=False, 
           bin_width=10, filter_fn='gaussian', 
           filter_kwargs={'sigma':1.5}),
    'S1': dict(boxcox=None, high_pass=False, bin_width=10, 
           filter_fn='gaussian', filter_kwargs={'sigma':1.5},
           region='S1'),
    'HPC_peanut': dict(spike_threshold=100, bin_width=10, 
                   speed_threshold=0, filter_fn='gaussian', 
                   filter_kwargs={'sigma':1.5}, boxcox=None),
    'ML': dict(spike_threshold=100, bin_width=5, 
               boxcox=None, filter_fn='gaussian', 
               filter_kwargs={'sigma':2.0}),
    'AM': dict(spike_threshold=100, bin_width=5, 
               boxcox=None, filter_fn='gaussian', 
               filter_kwargs={'sigma':2.0}),
    'VISp': dict(spike_threshold=100, bin_width=5, 
               boxcox=None, filter_fn='gaussian', 
               filter_kwargs={'sigma':2.0})
}

zlims = {
    'M1':[0, 4],
    'S1':[0, 4],
    'HPC_peanut':[0, 8],
    'ML': [0, 2],
    'AM': [0, 2],
    'VISp': [0, 2]
}

durations = {
    'M1': 100,
    'S1': 100,
    'HPC_peanut': 100,
    'ML': 100,
    'AM': 100,
    'VISp': 100
}

yticks = {
    'M1': [0, 50, 100],
    'S1': [0, 50, 100],
    'HPC_peanut': [0, 50, 100],
    'ML': [0, 50, 100],
    'AM': [0, 50, 100],
    'VISp': [0, 50, 100]
}

ytick_labels = {
    'M1': [0, 0.5, 1],
    'S1': [0, 0.5, 1],
    'HPC_peanut': [0, 0.5, 1],
    'ML': [0, 0.25, 0.5],
    'AM': [0, 0.25, 0.5],
    'VISp': [0, 0.25, 0.5]
}


# regions = ['M1', 'S1', 'HPC_peanut']
regions = ['VISp']
for region in regions:

    data_path = get_data_path(region)
    df, session_key = load_decoding_df(region, **loader_kwargs[region])

    sessions = np.unique(df[session_key].values)

    # For ML/AL, need this full_arg_tuple
    if 'full_arg_tuple' in df.keys():
        # Modify the full_arg_tuple according to the desired loader args
        full_arg_tuple = dict(df.iloc[0]['full_arg_tuple'])
        for k, v in rate_loader_kwargs[region].items():
            full_arg_tuple[k] = v
        full_arg_tuple = [tuple(full_arg_tuple.items())]
    else:
        full_arg_tuple = None
    # Use the first session (arbitrarily)
    sp_rates = get_rates_raw(data_path, region, sessions[0], 
                             loader_args=rate_loader_kwargs[region],
                             full_arg_tuple=full_arg_tuple)

    """fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    x_ = sp_rates[0][0:durations[region]]
    # sort by variancec
    var = np.var(x_, axis=0)
    #order = np.argsort(np.max(x_[idx], axis=0))
    #order = np.argsort(np.mean(x_[idx], axis=0))
    order = np.argsort(var)[::-1]
    #order = np.random.permutation(np.arange(x_.shape[1]))
    ax.view_init(elev=75, azim=0)

    # Transparent spines
    ax.grid(False)

    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    for j, i in enumerate(order[0:20]):
        # xx[xx <= -1] = np.nan
        ax.plot((2*j) * np.ones(x_.shape[0]), np.arange(x_.shape[0]), x_[:, i], 'k', alpha=0.5)
    ax.set_zlim(zlims[region])
    ax.set_zticks([])
    ax.set_xticks([])
    #ax.set_xlabel('Neurons')
    ax.set_yticks(yticks[region])
    ax.set_yticklabels(ytick_labels[region], fontsize=12)
    #ax.tick_params(axis='y', labelize=14)
    ax.set_ylabel('Time (s)', fontsize=14, labelpad=10)
    # ax.set_xlabel('Neurons', fontsize=14, labelpad=-300)
    # fig, ax = lt.subplots(1, 1, figsize=(5, 5))
    # ax.pcolor(x_[0].T, cmap='Greys')
    figpath = PATH_DICT['figs'] + '/rate_schematic_%s.pdf' % region
    fig.savefig(figpath, bbox_inches='tight', pad_inches=0)"""
    
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    x_ = sp_rates[0][0:durations[region]]

    # Sort by variance
    var = np.var(x_, axis=0)
    order = np.argsort(var)[::-1]

    ax.view_init(elev=75, azim=0)

    # Transparent spines (Matplotlib 3D axes fix)
    ax.grid(False)

    # Corrected way to make axis lines transparent
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Corrected way to make panes transparent
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))

    # Plot
    for j, i in enumerate(order[0:20]):
        ax.plot((2*j) * np.ones(x_.shape[0]), np.arange(x_.shape[0]), x_[:, i], 'k', alpha=0.5)

    ax.set_zlim(zlims[region])
    ax.set_zticks([])
    ax.set_xticks([])
    ax.set_yticks(yticks[region])
    ax.set_yticklabels(ytick_labels[region], fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=14, labelpad=10)

    # Save figure
    figpath = PATH_DICT['figs'] + f'/rate_schematic_{region}.pdf'
    fig.savefig(figpath, bbox_inches='tight', pad_inches=0)
        