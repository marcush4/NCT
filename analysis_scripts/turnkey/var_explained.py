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

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

import matplotlib.cm as cm
import matplotlib.colors as colors

try:
    from FCCA.fcca import FCCA as LQGCA
except:
    from FCCA.fcca import LQGComponentsAnalysis as LQGCA

from region_select import *
from config import PATH_DICT
sys.path.append(PATH_DICT['repo'])
from utils import calc_loadings, calc_cascaded_loadings


# Make a plot of the variance explained by FCCA vs PCA
# Switch the high-d dimreduc to not cross-validate...
if __name__ == '__main__':
    region = 'S1'

    if region not in ['M1', 'S1']:
        raise NotImplementedError
    
    if region == 'M1':
        with open(PATH_DICT['df'] + '/sabes_highd_dimreduc_df.pkl', 'rb') as f:
            rl = pickle.load(f)
    else:
        with open(PATH_DICT['df'] + '/sabes_highd_dimreducS1_df.pkl', 'rb') as f:
            rl = pickle.load(f)

    sabes_df = pd.DataFrame(rl)
    data_path = get_data_path(region)
    # What is the fraction of the asymptotic LQR cost/variance attained?
    dimvals = np.unique(sabes_df['dim'].values)
    data_files = np.unique(sabes_df['data_file'].values)

    if not os.path.exists(PATH_DICT['tmp'] + '/sabes_scores_tmp%s.pkl' % region):

        fcca_scores = np.zeros((dimvals.size, data_files.size, 2))
        pca_scores = np.zeros((dimvals.size, data_files.size, 2))
        fcca_ambient = np.zeros((data_files.size))

        for i, data_file in tqdm(enumerate(data_files)):
            dat = load_data(data_path, region, data_file, sabes_df.iloc[0]['loader_args'])
            X = np.squeeze(dat['spike_rates'])
            pcamodel = PCA().fit(X)
            pca_ambient = np.sum(pcamodel.explained_variance_)

            lqgmodel = LQGCA(T=3)
            lqgscore = lqgmodel.score(X=X, coef=np.eye(X.shape[1]))
            fcca_ambient[i] = lqgscore

            for j, d in enumerate(dimvals):
                df = apply_df_filters(sabes_df, data_file=data_file, dimreduc_method='LQGCA', dim=d, fold_idx=0)
                lqgmodel = LQGCA(T=3)
                lqgscore = lqgmodel.score(X=X, coef=df.iloc[0]['coef'])
                fcca_scores[j, i, 0] = lqgscore            
                lqgscore = lqgmodel.score(X=X, coef=pcamodel.components_.T[:, 0:d])
                fcca_scores[j, i, 1] = lqgscore
                pca_scores[j, i, 0] = np.sum(pcamodel.explained_variance_ratio_[0:d])
                C = np.cov(X @ df.iloc[0]['coef'], rowvar=False)
                if d > 1:
                    pca_scores[j, i, 1] = np.trace(np.cov(X @ df.iloc[0]['coef'], rowvar=False))/pca_ambient
                else:
                    pca_scores[j, i, 1] = np.var(X @ df.iloc[0]['coef'])/pca_ambient

        with open(PATH_DICT['tmp'] + '/sabes_scores_tmp%s.pkl' % region, 'wb') as f:
            f.write(pickle.dumps(fcca_scores))
            f.write(pickle.dumps(pca_scores))
            f.write(pickle.dumps(fcca_ambient))
    else:
        with open(PATH_DICT['tmp'] + '/sabes_scores_tmp%s.pkl' % region, 'rb') as f:
            fcca_scores = pickle.load(f)
            pca_scores = pickle.load(f)
            fcca_ambient = pickle.load(f)

    # Plot percentage of asymptotic score attained
    fcca_score_pcnt1 = (1 - np.divide(fcca_ambient - fcca_scores[..., 0], fcca_ambient - fcca_scores[0, :, 0]))
    fcca_score_pcnt2 = (1 - np.divide(fcca_ambient - fcca_scores[..., 1], fcca_ambient - fcca_scores[0, :, 1]))

    pca_score_pcnt1 = pca_scores[..., 0]
    pca_score_pcnt2 = pca_scores[..., 1]

    #fcca_score_normalized = np.divide(fcca_scores,fcca_ambient)
    #fcca_score_normalizedp = np.divide(fcca_scoresp,fcca_ambientp)
    #fcca_score_normalizedc = np.divide(fcca_scoresc,fcca_ambientc)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    dimvals = np.unique(sabes_df['dim'].values)
    ax.plot(dimvals, np.mean(fcca_score_pcnt1, axis=-1), 'r', alpha=0.75)
    ax.plot(dimvals, np.mean(fcca_score_pcnt2, axis=-1), 'r', linestyle='dashed', alpha=0.75)

    ax.fill_between(dimvals, np.mean(fcca_score_pcnt1, axis=-1) - np.std(fcca_score_pcnt1, axis=-1), 
                    np.mean(fcca_score_pcnt1, axis=-1) + np.std(fcca_score_pcnt1, axis=-1), color='r', alpha=0.25)
    ax.fill_between(dimvals, np.mean(fcca_score_pcnt2, axis=-1) - np.std(fcca_score_pcnt2, axis=-1), 
                    np.mean(fcca_score_pcnt2, axis=-1) + np.std(fcca_score_pcnt2, axis=-1), color='r', alpha=0.25)


    ax.plot(dimvals, np.mean(pca_score_pcnt1, axis=-1), 'k', alpha=0.75)
    ax.plot(dimvals, np.mean(pca_score_pcnt2, axis=-1), 'k', linestyle='dashed', alpha=0.75)

    ax.fill_between(dimvals, np.mean(pca_score_pcnt1, axis=-1) - np.std(pca_score_pcnt1, axis=-1), 
                    np.mean(pca_score_pcnt1, axis=-1) + np.std(pca_score_pcnt1, axis=-1), color='k', alpha=0.25)
    ax.fill_between(dimvals, np.mean(pca_score_pcnt2, axis=-1) - np.std(pca_score_pcnt2, axis=-1), 
                    np.mean(pca_score_pcnt2, axis=-1) + np.std(pca_score_pcnt2, axis=-1), color='k', alpha=0.25)

    ax.legend(['FBC, FCCA', 'FBC, PCA', 'FFC, PCA', 'FFC, FCCA'], loc='lower right')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Fraction of full dim. score attained')
    fig.tight_layout()
    fig.savefig(PATH_DICT['figs'] + '/%sscores.pdf' % region, bbox_inches='tight', pad_inches=0)