import numpy as np
import scipy 
import matplotlib.pyplot as plt
import sys, os
import pdb
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from dca.cov_util import form_lag_matrix

from pseudopy.nonnormal import NonnormalAuto, NonnormalPoints
from pseudopy.normal import Normal

from region_select import *
from config import PATH_DICT
sys.path.append(PATH_DICT['repo'])

from loaders import load_sabes, load_peanut, load_cv
from utils import form_companion, calc_loadings, apply_df_filters

regions = ['M1', 'S1', 'HPC_peanut']

ylim_dict = {
    'M1': [-0.8, 0.8],
    'S1': [-0.8, 0.8],
    'HPC_peanut': [-0.8, 0.8]
}

if __name__ == '__main__':
    for region in regions:
        if not os.path.exists(PATH_DICT['tmp'] + f'/{region}_pseudo_tmp.pkl'):
            data_path = get_data_path(region)
            df, session_key = load_decoding_df(region, **loader_kwargs[region])
            sessions = np.unique(df[session_key].values)

            Alist = []
            nnlist = []
            nlist = []

            for i, session in tqdm(enumerate(sessions)):
                dat = load_data(data_path, region, session, loader_args=df.iloc[0]['loader_args'])
                X = np.squeeze(dat['spike_rates'])
                xsmooth = scipy.ndimage.gaussian_filter1d(X, sigma=1, axis=0)

                xtrain = StandardScaler().fit_transform(xsmooth)
                xdiff_train = StandardScaler().fit_transform(np.diff(xsmooth, axis=0))
                linmodel = LinearRegression().fit(xtrain[:-1, :], xdiff_train)
                A_ = linmodel.coef_

                try:
                    nn1 = NonnormalAuto(A_, 1e-5, 1)
                    n1 = Normal(A_)
                except:
                    nn1 = None
                    n1 = None            

                Alist.append(A_)
                nnlist.append(nn1)
                nlist.append(n1)

            with open(PATH_DICT['tmp'] + f'/{region}_pseudo_tmp.pkl', 'wb') as f:
                f.write(pickle.dumps(Alist))
                f.write(pickle.dumps(nnlist))
                f.write(pickle.dumps(nlist))
                f.write(pickle.dumps(sessions))
        else:
            with open(PATH_DICT['tmp'] + f'/{region}_pseudo_tmp.pkl', 'rb') as f:
                Alist = pickle.load(f)
                nnlist = pickle.load(f)
                nlist = pickle.load(f)
                sessions = pickle.load(f)

        if not os.path.exists(PATH_DICT['figs'] + f'/{region}_pseudo'):
            os.makedirs(PATH_DICT['figs'] + f'/{region}_pseudo')

        for i, session in enumerate(sessions):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

            levels = [1e-6, 1e-1]
            A_ = Alist[i]
            nnobj = nnlist[i]
            nobj = nlist[i]
            if nnobj is None:
                continue

            ax.set_aspect('equal')
            ax.tricontourf(nnobj.triang, nnobj.vals, levels=levels, colors=['k'], alpha=0.2)

            ax.scatter(np.real(np.linalg.eigvals(A_)), np.imag(np.linalg.eigvals(A_)), 
                       s=15, alpha=0.5, marker='o', edgecolor='k', color='#6dd0ed')

            epsilons = list(np.sort(levels))
            padepsilons = [epsilons[0]*0.9] + epsilons + [epsilons[-1]*1.1]
            X = []
            Y = []
            Z = []
            for epsilon in padepsilons:
                paths = nobj.contour_paths(epsilon)
                for path in paths:
                    X += list(np.real(path.vertices[:-1]))
                    Y += list(np.imag(path.vertices[:-1]))
                    Z += [epsilon] * (len(path.vertices) - 1)
            ax.tricontour(X, Y, Z, levels=[1e-1], colors='k')


            # Add stability circle
            # circle1 = plt.Circle((0, 0), 1, color='k', fill=False, linestyle='--')
            # ax.add_patch(circle1)
            # ax.set_ylabel('Im' + r'$(z)$', fontsize=14)
            # ax.set_yticks([-1., -0.5, 0, 0.5, 1.])
            # ax.set_xlabel('Re' + r'$(z)$', fontsize=14)
            # ax.set_xticks([-1., -0.5, 0, 0.5, 1.])

            ax.set_xlim([-1, 0.2])
            ax.set_ylim(ylim_dict[region])
            #ax.xaxis.tick_top()
            #ax.xaxis.set_label_position('top')
            ax.spines['bottom'].set_position('zero')
            ax.spines['left'].set_position('zero')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_xticks([-1, 0])
            ax.set_yticks([-0.6, 0.6])

            ax.tick_params(axis='both', labelsize=14)
            fig.savefig(PATH_DICT['figs'] + f'/{region}_pseudo/{session}.pdf', bbox_inches='tight', pad_inches=0)
