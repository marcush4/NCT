import pdb
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.colors as colors

#sys.path.append('/home/akumar/nse/neural_control')
sys.path.append('/home/marcush/projects/neural_control/')
sys.path.append('/home/marcush/projects/neural_control/analysis_scripts/')
sys.path.append('/home/marcush/projects/github_repos')

from region_select import *
from utils import calc_loadings



def get_loadings_df(dimreduc_df, session_key, dim_select=6):

    # Load dimreduc_df and calculate loadings
    sessions = np.unique(dimreduc_df[session_key].values)
    nFolds = len(np.unique(dimreduc_df['fold_idx']))
    loadings_l = []

    for i, session in tqdm(enumerate(sessions)):
        loadings = []
        for dimreduc_method in [['LQGCA', 'FCCA'], 'PCA']:
            loadings_fold = [] 
            for fold_idx in range(nFolds):            
                df_filter = {session_key:session, 'fold_idx':fold_idx, 'dim':dim_select, 'dimreduc_method':dimreduc_method}
                df_ = apply_df_filters(dimreduc_df, **df_filter)

                assert(df_.shape[0] == 1)
                V = df_.iloc[0]['coef']
                if dimreduc_method == 'PCA':
                    V = V[:, 0:dim_select]        
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



def get_loadings(region, dimreduc_df, session_key, dim_select=21):

    if region in ['AM', 'ML']:
        dimreduc_reg_df = apply_df_filters(dimreduc_df, **{'loader_args':{'region': region}})
        loadings_df = get_loadings_df(dimreduc_reg_df, session_key, dim_select=21)
    else:
        loadings_df = get_loadings_df(dimreduc_df, session_key)


    # Add columns for relative importance
    loadings_df['rel_FFC'] = loadings_df['PCA_loadings'] / (loadings_df['PCA_loadings'] + loadings_df['FCCA_loadings'])
    loadings_df['rel_FBC'] = loadings_df['FCCA_loadings'] / (loadings_df['PCA_loadings'] + loadings_df['FCCA_loadings'])

    return loadings_df



def get_PSTHs(loadings_df, numUnitsPlot, data_path, region, session_key, zscore=False, full_arg_tuple=None):

    ####################### FFC Sorted
    key = 'rel_FFC'
    sorted_df = loadings_df.sort_values(by=key, ascending=False)
    top_n_rows = sorted_df.head(numUnitsPlot)
    uniq_data_files = np.unique(top_n_rows['data_file'])
    psths = []
    for session in uniq_data_files:
        top_n_rows_sess = apply_df_filters(top_n_rows, **{session_key:session})

        if region in ['AM', 'ML']:  
            x_ = get_psth(data_path, region, session, zscore=zscore, full_arg_tuple=dimreduc_df['full_arg_tuple'])
        else:
            x_ = get_psth(data_path, region, session, zscore=zscore)

        x = x_[top_n_rows_sess['nidx'].values, :]
        psths.append(x)
    ffc_psths = np.vstack(psths)

    ####################### FBC Sorted
    key = 'rel_FBC'
    sorted_df = loadings_df.sort_values(by=key, ascending=False)
    top_n_rows = sorted_df.head(numUnitsPlot)
    uniq_data_files = np.unique(top_n_rows['data_file'])
    psths = []
    for session in uniq_data_files:
        top_n_rows_sess = apply_df_filters(top_n_rows, **{session_key:session})

        if region in ['AM', 'ML']:  
            x_ = get_psth(data_path, region, session, zscore=zscore, full_arg_tuple=dimreduc_df['full_arg_tuple'])
        else:
            x_ = get_psth(data_path, region, session, zscore=zscore)

        x = x_[top_n_rows_sess['nidx'].values, :]
        psths.append(x)
    fbc_psths = np.vstack(psths)

    return ffc_psths, fbc_psths


def plot_PSTHs(ffc_psths, fbc_psths, bin_width, region, zscore=False, figpath='.'):
    
    ############################################ FFC
    fig, ax = plt.subplots(figsize=(5, 5))

    ffc_psths_smoothed = gaussian_filter1d(ffc_psths, sigma=2, axis=1)
    tSteps = ffc_psths.shape[1]
    time_vals = np.arange(0, tSteps*bin_width, bin_width)

    #pdb.set_trace()
    for psth in ffc_psths_smoothed:
        ax.plot(time_vals, psth, color='k')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))  # Place bottom spine at y=0
    ax.spines['bottom'].set_bounds(0, len(ffc_psths[0]))  # Only show positive x-axis
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_ylabel('Z-Scored Response' if zscore else 'Response', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.text(len(ffc_psths[0]), -0.05, 'Time (ms)', fontsize=18, ha='right')

   # ax.set_xlim(0, len(ffc_psths[0]))
    #ax.set_ylim(-0.5, 0.5)

    fig_save_path = f'{figpath}/score_sorted_PSTHs_{region}.pdf'
    fig.savefig(fig_save_path, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':

    region = 'AM'
    figpath = '/home/marcush/projects/neural_control/notebooks/TsaoLabNotebooks/CodeAndFigsForPaper/Figs'

    numUnitsPlot = 10
    dim_select = 21
    zscore = True

    dimreduc_df, session_key = load_decoding_df(region)
    bin_width = dimreduc_df['loader_args'][0]['bin_width']
    full_arg_tuple = dimreduc_df['full_arg_tuple'] if region in ['ML', 'AM'] else None

    data_path = get_data_path(region)
    loadings_df = get_loadings(region, dimreduc_df, session_key, dim_select=dim_select)
    ffc_psths, fbc_psths = get_PSTHs(loadings_df, numUnitsPlot, data_path, region, session_key, zscore=zscore, full_arg_tuple=full_arg_tuple)
    plot_PSTHs(ffc_psths, fbc_psths, bin_width, region, zscore=zscore, figpath=figpath)

dimreduc_df['loader_args']