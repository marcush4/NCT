#!/usr/bin/env python

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
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from config import *
import sys

sys.path.append(PATH_DICT['repo'])
from utils import apply_df_filters, calc_loadings, calc_cascaded_loadings
from loaders import load_sabes, load_peanut, load_cv
from decoders import lr_decoder, lr_encoder, logreg#, categorical_reg
from dca.cov_util import calc_cross_cov_mats_from_data
from region_select import *

#import dust

regions = ['VISp']
#regions = ['mPFC', "HPC"]

for region in regions:
    df, session_key = load_decoding_df(region, **loader_kwargs[region])
    data_path = get_data_path(region)
    sessions = np.unique(df[session_key].values)
    # if region == 'HPC_peanut':
    #     decoder_params = {'trainlag': 0, 'testlag': 0, 'decoding_window': 6}
    #     loader_args = {'bin_width':25, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}
    if region in ['mPFC','HPC']:
        decoder_params = {'trainlag': 1, 'testlag': 1, 'decoding_window': 6}
        loader_args = {'bin_width': 25, 'region': region, 'spike_threshold': 0, 'speed_threshold':False, 'trialize':False}
    else:
        decoder_params = df.iloc[0]['decoder_args']
        loader_args = df.iloc[0]['loader_args']
    su_l = []
    for i, session in tqdm(enumerate(sessions)):    

        if region in ['ML', 'AM']:
            dat = load_data(data_path, region, session, loader_args, df['full_arg_tuple'])
            sp_rates = dat['spike_rates']
            sp_mat = np.sum(sp_rates, 1).squeeze()
            X = sp_mat
            # depending on sklearn version..
            try:
                Z = OneHotEncoder(sparse_output=False).fit_transform(dat['StimIDs'].reshape(-1, 1))
            except:
                Z = OneHotEncoder(sparse=False).fit_transform(dat['StimIDs'].reshape(-1, 1))
        elif region in ['VISp']:
            
            unique_loader_args = list({frozenset(d.items()) for d in df['loader_args']})
            loader_args=dict(unique_loader_args[loader_kwargs[region]['load_idx']])
            dat = load_data(data_path, region, session, loader_args)    
            sp_rates = dat['spike_rates']       
            X =  np.sum(sp_rates, 1).squeeze()

            # depending on sklearn version..
            try:
                Z = OneHotEncoder(sparse_output=False).fit_transform(dat['behavior'].reshape(-1, 1))
            except:
                Z = OneHotEncoder(sparse=False).fit_transform(dat['behavior'].reshape(-1, 1))
            
        else:
            dat = load_data(data_path, region, session, loader_args)
            X = np.squeeze(dat['spike_rates'])
            Z = dat['behavior']

        kfold = KFold(n_splits=5, shuffle=False)

        # Average results across folds
        decoding_weights = []
        encoding_weights = []
        su_decoding_weights = []
        su_encoding_weights = []
        su_r2_pos = []
        su_r2_vel = []
        su_r2_enc = []

        su_decoding_r2 = []
        su_encoding_r2 = []

        # Single unit statistics
        su_var = np.zeros((5, X.shape[-1]))
        su_act = np.zeros((5, X.shape[-1]))

        for fold_idx, (train_idxs, test_idxs) in enumerate(kfold.split(X)):

            r = {}
            ztrain = Z[train_idxs]
            ztest = Z[test_idxs]

            # Population level decoding/encoding - use the coefficient in the linear fit
            # Record both the weights in the coefficient but also the loadings onto the SVD

            xtrain = X[train_idxs]
            xtest = X[test_idxs]

            if region in ['M1_maze']:
                ztrain = list(ztrain)
                ztest = list(ztest)
                xtrain = list(xtrain)
                xtest = list(xtest)


            if region in ['AM', 'ML', 'VISp']:
                ccm_xtrain = list_of_arrays = [sp_rates[i] for i in range(sp_rates.shape[0])] # just pass this as list of trials
                ccm = calc_cross_cov_mats_from_data(ccm_xtrain, T=sp_rates.shape[1]-1)
            elif region in ['M1_maze']:
                ccm_xtrain = list(X)
                ccm = calc_cross_cov_mats_from_data(ccm_xtrain, T=min([x.shape[0] for x in X]) - 1)
            else:
                ccm = calc_cross_cov_mats_from_data(xtrain, T=min([x.shape[0] for x in X]) - 1)

            ccm = torch.tensor(ccm)

            if region in ['ML', 'AM', 'VISp']:
                _, decodingregressor = logreg_decoder(xtest, xtrain, ztest, ztrain)
                _, encodingregressor = categorical_reg(xtest, xtrain, ztest, ztrain) # Use categorical regression for the encoder model
            else:
                _, _, _, decodingregressor, _, _, _ = lr_decoder(xtest, xtrain, ztest, ztrain, **decoder_params)
                _, encodingregressor = lr_encoder(xtest, xtrain, ztest, ztrain, **decoder_params)

            decoding_weights.append(decodingregressor.coef_)
            encoding_weights.append(encodingregressor.coef_)                
            
            r2_pos_decoding, r2_vel_decoding, r2_encoding = [], [], []
            
            su_dw = []
            su_ew = []            
            sur2pos = []
            sur2vel = []
            sur2enc = []

            # For ML/AM
            su_dec_r2 = [] # (McFadden's) R^2 value of predicting stimID from neural activity
            su_enc_r2 = [] #              R^2 value of predicting neural activity from stimID

            if region in ['M1_maze']:
                n_neurons = X[0].shape[-1]
            else:
                n_neurons = X.shape[-1]

            for neu_idx in range(n_neurons):           #Fit all neurons one by one

                if region in ['M1_maze']:
                    xtrain = [x_[:, neu_idx][:, np.newaxis] for x_ in X[train_idxs]]
                    xtest = [x_[:, neu_idx][:, np.newaxis] for x_ in X[test_idxs]]
                else:
                    xtrain = X[train_idxs, neu_idx][:, np.newaxis]
                    xtest = X[test_idxs, neu_idx][:, np.newaxis]

                if region in ['ML', 'AM', 'VISp']:
                    # Decoding
                    r2_decoding, decReg = logreg_decoder(xtest, xtrain, ztest, ztrain)
                    su_dw.append(decReg.coef_)
                    su_dec_r2.append(r2_decoding)

                    # Encoding
                    r2_encoding, encReg = categorical_reg(xtest, xtrain, ztest, ztrain)
                    su_ew.append(encReg.coef_)        
                    su_enc_r2.append(r2_encoding)
                else:
                    # Decoding
                    r2_pos, r2_vel, _, dr, _, _, _ = lr_decoder(xtest, xtrain, ztest, ztrain, **decoder_params)
                    r2_pos_decoding.append(r2_pos)
                    r2_vel_decoding.append(r2_vel)
                    su_dw.append(dr.coef_)
                    sur2pos.append(r2_pos)
                    sur2vel.append(r2_vel)

                    # Encoding
                    r2_encoding_, er = lr_encoder(xtest, xtrain, ztest, ztrain, **decoder_params)
                    r2_encoding.append(r2_encoding_)
                    su_ew.append(er.coef_)        
                    sur2enc.append(r2_encoding_)

            if region in ['ML', 'AM', 'VISp']:
                su_decoding_weights.append(np.array(su_dw))
                su_encoding_weights.append(np.array(su_ew))
                
                su_decoding_r2.append(np.array(su_dec_r2))
                su_encoding_r2.append(np.array(su_enc_r2))
            else:
                su_decoding_weights.append(np.array(su_dw))
                su_encoding_weights.append(np.array(su_ew))
                
                su_r2_pos.append(np.array(sur2pos))
                su_r2_vel.append(np.array(sur2vel))
                su_r2_enc.append(np.array(sur2enc))
            
            for neu_idx in range(n_neurons):

                if region in ['M1_maze']:
                    xtrain = [x_[:, neu_idx][:, np.newaxis] for x_ in X[train_idxs]]
                    xtest = [x_[:, neu_idx][:, np.newaxis] for x_ in X[test_idxs]]
                    xtrain = np.concatenate(xtrain)
                    xtest = np.concatenate(xtest)
                else:
                    xtrain = X[train_idxs, neu_idx][:, np.newaxis]
                    xtest = X[test_idxs, neu_idx][:, np.newaxis]

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

        # Average results across folds
        decoding_weights = np.mean(np.array(decoding_weights), axis=0)
        encoding_weights = np.mean(np.array(encoding_weights), axis=0)
        su_decoding_weights = np.mean(np.array(su_decoding_weights), axis=0)
        su_encoding_weights = np.mean(np.array(su_encoding_weights), axis=0)
        
        if region in ['ML', 'AM', 'VISp']:
            su_decoding_r2 = np.mean(np.array(su_decoding_r2), axis=0)
            su_encoding_r2 = np.mean(np.array(su_encoding_r2), axis=0)
        else:
            su_r2_pos = np.mean(np.array(su_r2_pos), axis=0)
            su_r2_vel = np.mean(np.array(su_r2_vel), axis=0)
            su_r2_enc = np.mean(np.array(su_r2_enc), axis=0)

        su_var = np.mean(su_var, axis=0)
        su_act = np.mean(su_act, axis=0)

        result = {}
        for variable in ('session', 'decoding_weights', 'encoding_weights', 
                        'su_decoding_weights', 'su_encoding_weights', 
                        'su_encoding_r2', 'su_decoding_r2',
                        'su_r2_pos', 'su_r2_vel', 'su_r2_enc', 
                        'su_var', 'su_act', 'decoder_params'):
            result[variable] = eval(variable)

        su_l.append(result)

    with open(PATH_DICT['tmp'] + '/su_calcs_%s.pkl' % region, 'wb') as f:
        f.write(pickle.dumps(su_l))
