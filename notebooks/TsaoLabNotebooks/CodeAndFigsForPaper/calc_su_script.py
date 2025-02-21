#!/usr/bin/env python3
import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import torch
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import os
import sys

sys.path.append('/home/marcush/projects/neural_control/')
sys.path.append('/home/marcush/projects/neural_control/analysis_scripts/')
sys.path.append('/home/marcush/projects/github_repos')
from utils import apply_df_filters, calc_loadings, calc_cascaded_loadings
from loaders import load_sabes, load_peanut, load_cv, load_tsao
from decoders import lr_decoder, lr_encoder, logreg_decoder, categorical_reg
from subspaces import SubspaceIdentification, IteratedStableEstimator, estimate_autocorrelation
from sklearn.preprocessing import OneHotEncoder

from dca.cov_util import calc_cross_cov_mats_from_data, calc_pi_from_cross_cov_mats
#from dca_research.kca import calc_mmse_from_cross_cov_mats
#from dca_research.lqg import build_loss as build_lqg_loss



# %% [markdown]
# # Replace this with a loaders call!

# %%
# Load Dataframe(s)
dataframe_path = '/home/marcush/Data/TsaoLabData/neural_control_output_new/decoding_deg_230809_140453_Alfie/decoding_deg_230809_140453_Alfie_glom.pickle'
savePath = os.path.dirname(dataframe_path)
with open(dataframe_path, 'rb') as f:
    rl = pickle.load(f)
tsao_df = pd.DataFrame(rl)

print(f"Working on: {dataframe_path}")

data_files = [tsao_df['data_file'][0]]
# Loop through the rest of this from here

# Load the spike rates
def make_hashable(obj):
    if isinstance(obj, dict):
        return tuple(sorted((key, make_hashable(value)) for key, value in obj.items()))
    elif isinstance(obj, (list, tuple)):
        return tuple(make_hashable(item) for item in obj)
    else:
        return obj


unique_hashes = set(make_hashable(d) for d in tsao_df['full_arg_tuple'])
unique_dicts = [dict(u) for u in unique_hashes]
preload_dict_path = tsao_df['data_path'][0] + "/preloaded/preloadDict.pickle"

with open(preload_dict_path, 'rb') as file:
    preloadDict = pickle.load(file)


for arg_dict in unique_dicts:
    arg_tuple = tuple(sorted(arg_dict.items()))


    for args in preloadDict.keys():

        if args == arg_tuple:

            preloadID = preloadDict[arg_tuple]
            loaded_data_path = os.path.dirname(preload_dict_path) + f"/preloaded_data_{preloadID}.pickle"
            
            if arg_dict['region'] == 'AM':
                with open(loaded_data_path, 'rb') as file:
                    AM_loaded_data = pickle.load(file)

            elif arg_dict['region'] == 'ML':
                with open(loaded_data_path, 'rb') as file:
                    ML_loaded_data = pickle.load(file)


AM_sp_rates = AM_loaded_data['spike_rates']
ML_sp_rates = ML_loaded_data['spike_rates']

AM_spike_mat = np.sum(AM_sp_rates, 1).squeeze()
ML_spike_mat = np.sum(ML_sp_rates, 1).squeeze()

# %%
region = "ML"



if region == 'AM':
    X = AM_spike_mat
    Z = OneHotEncoder(sparse_output=False).fit_transform(AM_loaded_data['StimIDs'].reshape(-1, 1))
elif region == 'ML':
    X = ML_spike_mat
    Z = OneHotEncoder(sparse_output=False).fit_transform(ML_loaded_data['StimIDs'].reshape(-1, 1))
# Make targets one-hot since categorical

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=False)
decoder_params = tsao_df['decoder_args'][0]

tsao_results = []

# Average results across folds
decoding_weights = []
encoding_weights = []

su_decoding_weights = []
su_encoding_weights = []

su_decoding_r2 = []
su_encoding_r2 = []

# Single unit statistics
su_var = np.zeros((n_splits, X.shape[-1]))
su_act = np.zeros((n_splits, X.shape[-1]))


# %%
for i, data_file in tqdm(enumerate(data_files)):    

    for fold_idx, (train_idxs, test_idxs) in enumerate(kfold.split(X)):

        r = {}

        ztrain = Z[train_idxs, :]
        ztest = Z[test_idxs, :]

        xtrain = X[train_idxs, :]
        xtest = X[test_idxs, :]

        # Use time resolved responses to determine unit covariances, and use as much time as possible to estimate this.
        if region == 'AM':
            ccm_xtrain = list_of_arrays = [AM_sp_rates[i] for i in range(AM_sp_rates.shape[0])] # just pass this as list of trials
            ccm = calc_cross_cov_mats_from_data(ccm_xtrain, T=AM_sp_rates.shape[1]-1)
        elif region == 'ML':
            ccm_xtrain = list_of_arrays = [ML_sp_rates[i] for i in range(ML_sp_rates.shape[0])] # just pass this as list of trials
            ccm = calc_cross_cov_mats_from_data(ccm_xtrain, T=ML_sp_rates.shape[1]-1)
        
        ccm = torch.tensor(ccm)

        _, decodingregressor = logreg_decoder(xtest, xtrain, ztest, ztrain)
        _, encodingregressor = categorical_reg(xtest, xtrain, ztest, ztrain) # Use categorical regression for the encoder model

        decoding_weights.append(decodingregressor.coef_)
        encoding_weights.append(encodingregressor.coef_)                
        
        
        su_dw = []     # Single Unit Decoding Weights
        su_ew = []     # Single Unit Encoding Weights
        su_dec_r2 = [] # (McFadden's) R^2 value of predicting stimID from neural activity
        su_enc_r2 = [] #              R^2 value of predicting neural activity from stimID

        for neu_idx in range(X.shape[-1]):           #Fit all neurons one by one
            
            xtrain = X[train_idxs, neu_idx][:, np.newaxis]
            xtest = X[test_idxs, neu_idx][:, np.newaxis]

            # Decoding
            r2_decoding, decReg = logreg_decoder(xtest, xtrain, ztest, ztrain)
            su_dw.append(decReg.coef_)
            su_dec_r2.append(r2_decoding)

            # Encoding
            r2_encoding, encReg = categorical_reg(xtest, xtrain, ztest, ztrain)
            su_ew.append(encReg.coef_)        
            su_enc_r2.append(r2_encoding)


        su_decoding_weights.append(np.array(su_dw))
        su_encoding_weights.append(np.array(su_ew))
        
        su_decoding_r2.append(np.array(su_dec_r2))
        su_encoding_r2.append(np.array(su_enc_r2))

        
        for neu_idx in range(X.shape[-1]):

            xtrain = X[train_idxs, neu_idx][:, np.newaxis]
            su_var[fold_idx, neu_idx] = np.var(xtrain)
            
            ccm_j = ccm[:, neu_idx, neu_idx].numpy()
            ccm_j /= ccm_j[0]

            thr = 1e-1
            acov_crossing = np.where(ccm_j < thr)
            if len(acov_crossing[0]) > 0:
                su_act[fold_idx, neu_idx] = np.where(ccm_j < thr)[0][0]
            else:
                su_act[fold_idx, neu_idx] = len(ccm)


        # Calculate decoding weights based on projection of the data first
        print(f"Done with {fold_idx+1} fold")



# %%
# Average results across folds and save
decoding_weights = np.mean(np.array(decoding_weights), axis=0)
encoding_weights = np.mean(np.array(encoding_weights), axis=0)
su_decoding_weights = np.mean(np.array(su_decoding_weights), axis=0)
su_encoding_weights = np.mean(np.array(su_encoding_weights), axis=0)

su_decoding_r2 = np.mean(np.array(su_decoding_r2), axis=0)
su_encoding_r2 = np.mean(np.array(su_encoding_r2), axis=0)

su_var = np.mean(su_var, axis=0)
su_act = np.mean(su_act, axis=0)

result = {}
for variable in ('data_file', 'decoding_weights', 'encoding_weights', 'su_decoding_weights', 'su_encoding_weights', 'su_decoding_r2'
                 , 'su_encoding_r2', 'su_var', 'su_act', 'decoder_params'):
    result[variable] = eval(variable)

tsao_results.append(result)

with open(f'{savePath}/tsao_su_calcs_{region}.dat', 'wb') as f:
    f.write(pickle.dumps(tsao_results))

# %% [markdown]
# # Region AM

# %%


# %%
region = "AM"


if region == 'AM':
    X = AM_spike_mat
    Z = OneHotEncoder(sparse_output=False).fit_transform(AM_loaded_data['StimIDs'].reshape(-1, 1))
elif region == 'ML':
    X = ML_spike_mat
    Z = OneHotEncoder(sparse_output=False).fit_transform(ML_loaded_data['StimIDs'].reshape(-1, 1))
# Make targets one-hot since categorical

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=False)
decoder_params = tsao_df['decoder_args'][0]

tsao_results = []

# Average results across folds
decoding_weights = []
encoding_weights = []

su_decoding_weights = []
su_encoding_weights = []

su_decoding_r2 = []
su_encoding_r2 = []

# Single unit statistics
su_var = np.zeros((n_splits, X.shape[-1]))
su_act = np.zeros((n_splits, X.shape[-1]))



for i, data_file in tqdm(enumerate(data_files)):    

    for fold_idx, (train_idxs, test_idxs) in enumerate(kfold.split(X)):

        r = {}

        ztrain = Z[train_idxs, :]
        ztest = Z[test_idxs, :]

        xtrain = X[train_idxs, :]
        xtest = X[test_idxs, :]

        # Use time resolved responses to determine unit covariances, and use as much time as possible to estimate this.
        if region == 'AM':
            ccm_xtrain = list_of_arrays = [AM_sp_rates[i] for i in range(AM_sp_rates.shape[0])] # just pass this as list of trials
            ccm = calc_cross_cov_mats_from_data(ccm_xtrain, T=AM_sp_rates.shape[1]-1)
        elif region == 'ML':
            ccm_xtrain = list_of_arrays = [ML_sp_rates[i] for i in range(ML_sp_rates.shape[0])] # just pass this as list of trials
            ccm = calc_cross_cov_mats_from_data(ccm_xtrain, T=ML_sp_rates.shape[1]-1)
        
        ccm = torch.tensor(ccm)

        _, decodingregressor = logreg_decoder(xtest, xtrain, ztest, ztrain)
        _, encodingregressor = categorical_reg(xtest, xtrain, ztest, ztrain) # Use categorical regression for the encoder model

        decoding_weights.append(decodingregressor.coef_)
        encoding_weights.append(encodingregressor.coef_)                
        
        
        su_dw = []     # Single Unit Decoding Weights
        su_ew = []     # Single Unit Encoding Weights
        su_dec_r2 = [] # (McFadden's) R^2 value of predicting stimID from neural activity
        su_enc_r2 = [] #              R^2 value of predicting neural activity from stimID

        for neu_idx in range(X.shape[-1]):           #Fit all neurons one by one
            
            xtrain = X[train_idxs, neu_idx][:, np.newaxis]
            xtest = X[test_idxs, neu_idx][:, np.newaxis]

            # Decoding
            r2_decoding, decReg = logreg_decoder(xtest, xtrain, ztest, ztrain)
            su_dw.append(decReg.coef_)
            su_dec_r2.append(r2_decoding)

            # Encoding
            r2_encoding, encReg = categorical_reg(xtest, xtrain, ztest, ztrain)
            su_ew.append(encReg.coef_)        
            su_enc_r2.append(r2_encoding)


        su_decoding_weights.append(np.array(su_dw))
        su_encoding_weights.append(np.array(su_ew))
        
        su_decoding_r2.append(np.array(su_dec_r2))
        su_encoding_r2.append(np.array(su_enc_r2))

        
        for neu_idx in range(X.shape[-1]):

            xtrain = X[train_idxs, neu_idx][:, np.newaxis]
            su_var[fold_idx, neu_idx] = np.var(xtrain)
            
            ccm_j = ccm[:, neu_idx, neu_idx].numpy()
            ccm_j /= ccm_j[0]

            thr = 1e-1
            acov_crossing = np.where(ccm_j < thr)
            if len(acov_crossing[0]) > 0:
                su_act[fold_idx, neu_idx] = np.where(ccm_j < thr)[0][0]
            else:
                su_act[fold_idx, neu_idx] = len(ccm)


        # Calculate decoding weights based on projection of the data first
        print(f"Done with {fold_idx+1} fold")



# Average results across folds and save
decoding_weights = np.mean(np.array(decoding_weights), axis=0)
encoding_weights = np.mean(np.array(encoding_weights), axis=0)
su_decoding_weights = np.mean(np.array(su_decoding_weights), axis=0)
su_encoding_weights = np.mean(np.array(su_encoding_weights), axis=0)

su_decoding_r2 = np.mean(np.array(su_decoding_r2), axis=0)
su_encoding_r2 = np.mean(np.array(su_encoding_r2), axis=0)

su_var = np.mean(su_var, axis=0)
su_act = np.mean(su_act, axis=0)

result = {}
for variable in ('data_file', 'decoding_weights', 'encoding_weights', 'su_decoding_weights', 'su_encoding_weights', 'su_decoding_r2'
                 , 'su_encoding_r2', 'su_var', 'su_act', 'decoder_params'):
    result[variable] = eval(variable)

tsao_results.append(result)

with open(f'{savePath}/tsao_su_calcs_{region}.dat', 'wb') as f:
    f.write(pickle.dumps(tsao_results))


