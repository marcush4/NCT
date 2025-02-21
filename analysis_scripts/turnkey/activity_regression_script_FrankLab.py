#!/usr/bin/env python

import argparse
import pickle
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import glob as glob
import os
from config import PATH_DICT
import sys
sys.path.append(PATH_DICT['repo'])
from region_select import *
from utils import calc_loadings
from collections import defaultdict
from scipy.linalg import subspace_angles
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from loaders import load_tsao

dim_dict = {
    'M1': 6,
    'M1_trialized':6,
    'S1': 6,
    'HPC_peanut': 6,
    'M1_maze': 6,
    'AM': 21,
    'ML': 21,
    'HPC':15,
    'mPFC':15
}

if __name__ == "__main__":


    """" 
    ----------------------------------------------------------------------------------------------------------------
    Handful of Hardcoded Values: 
    1. the output name to save the projected data R^2 values
    2. The name of the dataframe (could also use get_df)
    3. The CCA dimensionality that you want to use for each recording session.  E.g.: manual_dimensions = [21, 21] 
    ----------------------------------------------------------------------------------------------------------------
    """ 
    
    comparisons_save_path = f"{PATH_DICT['tmp']}/activity_regression_scores_franklab.pkl"
    df_name = '/decoding_fullarg_frank_lab_glom.pickle'
    manual_dimensions = [20] 



    cca_base_path = '/clusterfs/NSDS_data/FCCA/postprocessed/CCA_structs/'
    figpath = PATH_DICT['figs']
    root_path = PATH_DICT['df']
    
        
    RELOAD = True
    
    
    print("Loading decode dataframe ..")

    with open(root_path + df_name, 'rb') as f:
        rl = pickle.load(f)
    df_decode = pd.DataFrame(rl)
    
    
    n_folds = np.unique(df_decode['fold_idx'])
    dimreduc_methods = np.unique(df_decode['dimreduc_method'])
    proj_methods = np.append(dimreduc_methods, 'CCA')
    #regions = np.unique(df_decode['loader_args'].apply(lambda x: x.get('region')))
    regions = ['mPFC', 'HPC'] # unique will put this in alph. order. Want ML first for later plots.
    sessions = np.unique(df_decode['data_file'])
    
    
    if RELOAD:
            

        print("Loading spike rates ..")
        all_spikes = {}
        for session in sessions:
            all_spikes[session] = {}
            for region in regions: 
            
                
                load_idx, dec_idx, dr_idx = loader_kwargs[region].values()
                loader_args, decoder_args, dimreduc_args = get_franklab_args(load_idx, dec_idx, dr_idx)                
                          
                df_ = apply_df_filters(df_decode, **{'loader_args':{'region': region}})      
                df_PCA = apply_df_filters(df_, loader_args=loader_args, decoder_args=decoder_args, dimreduc_method="PCA")
                df_LQGCA = apply_df_filters(df_, loader_args=loader_args, decoder_args=decoder_args, dimreduc_args=dimreduc_args)        
                df_ = pd.concat([df_PCA, df_LQGCA])       
                                  
                data_path = get_data_path(region)
                dat = load_data(data_path, region, session, loader_args)                
                spikes =  dat['spike_rates']  
                all_spikes[session][region] =  spikes.reshape(-1, spikes.shape[-1])
            

        print("Loading CCA Models ..")
        cca_models = {}
        for sessIdx, session in enumerate(sessions):
            
            cca_path = f'{cca_base_path}CCA_{session}_{manual_dimensions[sessIdx]}_dims.pkl'
            with open(cca_path, 'rb') as file:
                ccamodel = pickle.load(file)
            cca_models[session] = ccamodel
            
        
        
        all_comparisons_0_1 = []
        all_comparisons_1_0 = []
        proj_correlations = {}


        print("Beginning Main Iter: ")
        combo_ind = 0
        region_method_pairs = set()  # Set to keep track of seen combinations
        for region0, region1, method0, method1 in itertools.product(regions, regions, proj_methods, proj_methods):
            
            print(f"Starting outer loop {region0}, {region1}, {method0}, {method1}")
            
            sorted_0 = tuple(sorted([region0, method0]))
            sorted_1 = tuple(sorted([region1, method1]))
            
            combo = (tuple(sorted((sorted_0, sorted_1))))
            
            if combo in region_method_pairs:
                continue  
            elif region0 == region1 and method0 == method1:
                continue
            region_method_pairs.add(combo)
            

            n_fold_r2_0_1 = np.zeros((len(n_folds), len(sessions)))
            n_fold_r2_1_0 = np.zeros((len(n_folds), len(sessions)))
            avg_correlations = np.zeros((len(n_folds), len(sessions)))
            
            
            for n_fold in n_folds:
                print(f"-- starting fold {n_fold}")
                for sessInd, session in enumerate(sessions): 
                    print(f"-- starting session {sessInd}")

                    
                    DIM = manual_dimensions[np.where(session == sessions)[0][0]]
                    ccamodel = cca_models[session]


                    if method0 == 'CCA' and method1 == 'CCA':
                        if region0 == regions[0]:
                            ProjMat0 = ccamodel.x_rotations_
                            ProjMat1 = ccamodel.y_rotations_
                        elif region0 == regions[1]:
                            ProjMat0 = ccamodel.y_rotations_
                            ProjMat1 = ccamodel.x_rotations_

                    elif method0 == 'CCA':
                        ProjMat1 =  apply_df_filters(df_decode, **{'fold_idx':n_fold, 'dim':DIM, 'dimreduc_method':method1, 'loader_args':{'region':region1}})['coef'].iloc[0]

                        if region0 == regions[0]:
                            ProjMat0 = ccamodel.x_rotations_
                        elif region0 == regions[1]:
                            ProjMat0 = ccamodel.y_rotations_

                    elif method1 == 'CCA':
                        ProjMat0 =  apply_df_filters(df_decode, **{'fold_idx':n_fold, 'dim':DIM, 'dimreduc_method':method0, 'loader_args':{'region':region0}})['coef'].iloc[0]
                        if region1 == regions[0]:
                            ProjMat1 = ccamodel.x_rotations_
                        elif region1 == regions[1]:
                            ProjMat1 = ccamodel.y_rotations_
                    else:

                        ProjMat0 =  apply_df_filters(df_decode, **{'fold_idx':n_fold, 'dim':DIM, 'dimreduc_method':method0, 'loader_args':{'region':region0}})['coef'].iloc[0]
                        ProjMat1 =  apply_df_filters(df_decode, **{'fold_idx':n_fold, 'dim':DIM, 'dimreduc_method':method1, 'loader_args':{'region':region1}})['coef'].iloc[0]


                    projData0 = all_spikes[session][region0] @ ProjMat0[:,0:DIM]
                    projData1 = all_spikes[session][region1] @ ProjMat1[:,0:DIM]


                    model_0_1 = LinearRegression()
                    model_0_1.fit(projData0, projData1)
                    r2_0_1 = model_0_1.score(projData0, projData1)
                    
                    model_1_0 = LinearRegression()
                    model_1_0.fit(projData1, projData0)
                    r2_1_0 = model_1_0.score(projData1, projData0)

                    correlations = []
                    for row_x, row_y in zip(projData0, projData1):
                        corr, _ = pearsonr(row_x, row_y)
                        correlations.append(corr)
                    average_correlation = np.mean(correlations)


                    avg_correlations[n_fold, sessInd] = average_correlation
                    n_fold_r2_0_1[n_fold, sessInd] = r2_0_1
                    n_fold_r2_1_0[n_fold, sessInd] = r2_1_0

                
            R2_avg_0_1 = np.mean(n_fold_r2_0_1)
            R2_avg_1_0 = np.mean(n_fold_r2_1_0)

            all_comparisons_0_1.append(R2_avg_0_1)
            all_comparisons_1_0.append(R2_avg_1_0)

            proj_correlations[f"{region0}_{method0}_{region1}_{method1}"] = np.mean(avg_correlations)
                
            combo_ind += 1


        ###################### Save Main Result ######################

        comparisons = {"Region0_to_Region1":all_comparisons_0_1, "Region1_to_Region0":all_comparisons_1_0}            
        with open(comparisons_save_path, 'wb') as f:
            pickle.dump(comparisons, f)
        
        
    else:        
        with open(comparisons_save_path, 'rb') as f:
            comparisons = pickle.load(f)
            
        all_comparisons_0_1 = comparisons['Region0_to_Region1']
        all_comparisons_1_0 = comparisons['Region1_to_Region0']
        
        
        
    
    ###################### Plot Main Figure ######################
    unique_groups = [(regions[0], 'CCA'), (regions[0], 'LQGCA'), (regions[0], 'PCA'), (regions[1], 'CCA'), (regions[1], 'LQGCA'), (regions[1], 'PCA')]
    unique_group_labels = [(regions[0], 'CCA'), (regions[0], 'FBC'), (regions[0], 'FFC'), (regions[1], 'CCA'), (regions[1], 'FBC'), (regions[1], 'FFC')]

    n = 6  # We have 6 unique groups
    matrix = np.zeros((n, n))  # Initialize the matrix with zeros

    correlation_index = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = all_comparisons_0_1[correlation_index]
            matrix[j, i] = all_comparisons_1_0[correlation_index]  # Not Symmetric
            correlation_index += 1

    # Set diagonal values to NaN
    np.fill_diagonal(matrix, np.nan)

    labels = [f"{reg}, {meth}" for reg, meth in unique_group_labels]

    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='gray_r', vmin=0, vmax=1)
    plt.colorbar(cax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticklabels(labels)

    # Annotate each cell with the numeric value
    for (i, j), val in np.ndenumerate(matrix):
        if not np.isnan(val):  # Skip NaN values
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if abs(val) > 0.5 else 'black')

    midpoint = n / 2
    ax.axhline(midpoint - 0.5, color='black', linewidth=0.5)  # Horizontal line
    ax.axvline(midpoint - 0.5, color='black', linewidth=0.5)  # Vertical line

    ax.set_title(f'R² value between projected activity')
    ax.set_xlabel('region, subspace')
    ax.set_ylabel('region, subspace')


    figpath = PATH_DICT['figs'] + '/activity_regression_total_%s.pdf' % 'franklab'
    fig.savefig(figpath, bbox_inches='tight', pad_inches=0)
    
    
    
    
    
    ###################### Plot Sub Figures ######################
    submatrices = [  matrix[:3, :3],  matrix[:3, 3:], matrix[3:, :3],  matrix[3:, 3:]]
    sub_labels = [labels[:3],labels[:3],  labels[3:],labels[3:]]
    

    titles = ["Within Region ML", "Cross Regions ML-AM", "Cross Regions AM-ML", "Within Region AM"]
    for i in range(4):
        
        max_val = np.nanmax(submatrices[i])

        fig, ax = plt.subplots(figsize=(4.5, 4))
        cax = ax.matshow(submatrices[i], cmap='gray_r', vmin=0, vmax=max_val)
        fig.colorbar(cax, ax=ax)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))

        if i == 1:  # Cross Regions AM-ML
            ax.set_xticklabels(labels[3:], rotation=45, ha="left", rotation_mode="anchor")
            ax.set_yticklabels(labels[:3])
        elif i == 2:  # Cross Regions ML-AM
            ax.set_xticklabels(labels[:3], rotation=45, ha="left", rotation_mode="anchor")
            ax.set_yticklabels(labels[3:])
        else:  # Within Region plots
            ax.set_xticklabels(sub_labels[i], rotation=45, ha="left", rotation_mode="anchor")
            ax.set_yticklabels(sub_labels[i])

        for (x, y), val in np.ndenumerate(submatrices[i]):
            if not np.isnan(val):  # Skip NaN values
                ax.text(y, x, f"{val:.2f}", ha='center', va='center', color='white' if abs(val) >= 0.4 else 'black')


        ax.set_title(titles[i])
        ax.set_xlabel('region, subspace')
        ax.set_ylabel('region, subspace')
                        
        figpath = f"{PATH_DICT['figs']}/activity_regression_franklab_subplot_{i}.pdf"
        fig.savefig(figpath, bbox_inches='tight', pad_inches=0)


    