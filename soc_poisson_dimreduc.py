import itertools
import os
import time
import numpy as np
import scipy
import torch
import sys
import sdeint
import pickle
import joblib
from glob import glob
from tqdm import tqdm
from dca.methods_comparison import JPCA
from sklearn.decomposition import PCA
from dca.cov_util import calc_cross_cov_mats_from_data

#from dca_research.lqg import LQGComponentsAnalysis as LQGCA
from FCCA.fcca import LQGComponentsAnalysis as LQGCA
from soc import stabilize, gen_init_W, stabilize_discrete, comm_mat

from mpi4py import MPI

<<<<<<< Updated upstream
data_path = '/mnt/Secondary/data/soc_poisson2_100'
save_path = '/mnt/Secondary/data/soc_poisson2_100_dimreduc'
=======
#data_path = '/mnt/Secondary/data/soc_poisson2_100'
#save_path = '/mnt/Secondary/data/soc_poisson2_100_dimreduc'
data_path = '/home/ankit_kumar/Data/soc_poisson_sdeint2'
save_path = '/home/ankit_kumar/Data/soc_poisson_sdeint_dimreduc2'

if not os.path.exists(save_path):
    os.makedirs(save_path)

>>>>>>> Stashed changes
dims = [2, 3, 4, 5, 6]
T = [2, 3, 4]

fls = glob('%s/*.pkl' % data_path)

def inner_loop(fl):
    t0 = time.time()
    rl = []
    with open(fl, 'rb') as f:
        while True:
            try:
                task = pickle.load(f)
                rep, inner_rep, r = task
                A = pickle.load(f)
                nn = pickle.load(f)
                _ = pickle.load(f)
                cross_covs = pickle.load(f)
                cross_covs_rev = [np.linalg.inv(cross_covs[0]) @ c.T @ np.linalg.inv(cross_covs[0]) for c in cross_covs]
                cross_covs = torch.tensor(cross_covs)
                cross_covs_rev = torch.tensor(cross_covs_rev)
                e, Upca = np.linalg.eig(cross_covs[0])
                eigorder = np.argsort(e)[::-1]

                for d in dims:
                    pca_coef = Upca[:, eigorder][:, 0:d]
                    for T_ in T:
                        rd = {}
                        lqgmodel = LQGCA(d=d, T=T_, rng_or_seed=int(inner_rep))
                        lqgmodel.cross_covs = cross_covs
                        lqgmodel.cross_covs_rev = cross_covs_rev
                        coef_, score = lqgmodel._fit_projection()
                        rd['rep'] = int(rep)
                        rd['inner_rep'] = int(inner_rep)
                        rd['R'] = r
                        rd['pca_coef'] = pca_coef
                        rd['fcca_coef'] = coef_
                        rd['fcca_score'] = score
                        rd['pca_eig'] = e[eigorder][0:d]
                        rd['dim'] = d
                        rd['T'] = T_
                        rd['nn'] = nn
                        rl.append(rd)
            except EOFError:
                break
    fname = fl.split('.pkl')[0].split('/')[-1]
    fname += '_dr.pkl'
    with open(save_path + '/' + fname, 'wb') as f:
        f.write(pickle.dumps(rl))
    print('Finished %s in %f' % (fl, time.time() - t0))

if __name__ == '__main__':
    joblib.Parallel(n_jobs=24)(joblib.delayed(inner_loop)(fl) for fl in fls)
