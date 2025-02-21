import itertools
import time
import os
import numpy as np
import scipy
import torch
import sys
import sdeint
import pickle
from tqdm import tqdm
from dca.methods_comparison import JPCA
from dca.cov_util import calc_cross_cov_mats_from_data

#from dca_research.lqg import LQGComponentsAnalysis as LQGCA
from FCCA.fcca import LQGComponentsAnalysis as LQGCA
from soc import stabilize, gen_init_W, stabilize_discrete, comm_mat

from mpi4py import MPI

reps = 20
inner_reps = 10
M = 100
p = 0.25
    #gamma = np.array([2])
g = 2
R = np.linspace(0.75, 20, 25)
<<<<<<< Updated upstream
R = np.concatenate([R, np.array([11, 12, 13, 14, 15])]) 

=======
 
>>>>>>> Stashed changes
# For non-stationarity, choose a sequence of A matrices
# and switch between them at uniform intervals

# Use the rep parameter to choose the sequence.
def gen_activity_ns(tau, Wseq, activ_func, sigma, T, h):

    # Apply a slow unitary rotation to W - this preserves spectrum and
    # non-normality
    #interp_intervals = np.linspace(0, T, len(Wseq))  
    #interval_widths = np.diff(interp_intervals)[0]

    def Wt(t):
        return Wseq[0]
    # f
    def f_(x, t):
        W = Wt(t)
        return 1/tau * (-1 * np.eye(W.shape[0]) @ x + W @ activ_func(x))

    # G: linear i.i.d noise with sigma
    def g_(x, t):
        return sigma * np.eye(Wseq[0].shape[0])

    # Generate random initial condition and then integrate over the desired time period
    tspace = np.linspace(0, T, int(T/h))
    x0 = np.random.normal(size=(Wseq[0].shape[0],))
    return  sdeint.itoSRI2(f_, g_, x0, tspace)    

def gen_matrices():

    Alist = []
    for i in tqdm(range(reps)):
        Alist.append([])
        for j, r in enumerate(R):
            A = gen_init_W(M, p, g, r, -1)
            eig = np.linalg.eigvals(A)
            if np.max(np.real(eig)) >= 0:
                A = stabilize(A)
                eig = np.linalg.eigvals(A)

            assert(np.max(np.real(eig)) < 0)

            Alist[i].append(A)

    with open('soc_jpca_Atmp.pkl', 'wb') as f:
        f.write(pickle.dumps(Alist))
 
if __name__ == '__main__':

    # gen_matrices()
    # print('generated!')

    comm = MPI.COMM_WORLD
<<<<<<< Updated upstream
    activitypath = sys.argv[1]
    savepath = sys.argv[2]
    assert(os.path.exists(savepath))
    d =2
    T=3
=======
    savepath = sys.argv[1]
    assert(os.path.exists(savepath))
    d = 6
    T=4
>>>>>>> Stashed changes

    tasks = list(itertools.product(np.arange(reps), np.arange(inner_reps), R))
    tasks = np.array_split(tasks, comm.size)[comm.rank]
    
    #phi = np.zeros(len(tasks))
    #scores = np.zeros(len(tasks))
    #nn = np.zeros((len(tasks), 2))
    #jpca_eig = np.zeros((len(tasks), 2))

    # #Alist = []

    # First load generated A matrics
    if comm.rank == 0:
        with open('soc_Alist.pkl', 'rb') as f:
            Alist = pickle.load(f)
    else:
        Alist = None

    Alist = comm.bcast(Alist)
    
    # Select sequences of A matrices from the Alist to take the 
    
    print(len(tasks))

    for i, task in enumerate(tasks):
        t0 = time.time()
        rep, inner_rep, r = task
        rep = int(rep)
        inner_rep = int(inner_rep)
        ridx = list(R).index(r)
        rnd = np.random.RandomState(rep)
        seq_idxs = rnd.choice(np.arange(reps), 3)
        
        A = Alist[rep][ridx]
        nn = np.linalg.norm(A @ A.T - A.T @ A)
        #Alist.append(A)
        # Simulate from the model
<<<<<<< Updated upstream
        #x = gen_activity_ns(1, Aseq, lambda x: x, 1, 1000, 1e-1) 
        #print('activity gen')

        # Load the activity for the task
        with open('%s/x%d_%d_%d.pkl' % (activitypath, rep, r, inner_rep), 'rb') as f:
            x = pickle.load(f)


        cross_covs = calc_cross_cov_mats_from_data(x, 5)        
        # Fit PCA/FCCA
        cross_covs_rev = [np.linalg.inv(cross_covs[0]) @ c.T @ np.linalg.inv(cross_covs[0]) for c in cross_covs]
        cross_covs = torch.tensor(cross_covs)
        cross_covs_rev = torch.tensor(cross_covs_rev)
        e, Upca = np.linalg.eig(cross_covs[0])
        eigorder = np.argsort(e)[::-1]
        pca_coef = Upca[:, eigorder][:, 0:d]
        lqgmodel = LQGCA(d=d, T=T, rng_or_seed=int(inner_rep))
        lqgmodel.cross_covs = cross_covs
        lqgmodel.cross_covs_rev = cross_covs_rev
        coef_, score = lqgmodel._fit_projection()

        rd = {}
        rd['rep'] = rep
        rd['inner_rep'] = inner_rep
        rd['R'] = r
        rd['pca_coef'] = pca_coef
        rd['fcca_coef'] = coef_
        rd['fcca_score'] = score
        rd['pca_eig'] = e[eigorder][0:d]
        rd['dim'] = d
        rd['T'] = T
        rd['nn'] = nn  
    
=======
        x = gen_activity_ns(1, Aseq, lambda x: x, 1, 1000, 1e-1) 
        print('activity gen')

        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print('Unstable simulation')
            cross_covs = np.nan
            rd = {}
        else:
            try:
                cross_covs = calc_cross_cov_mats_from_data(x, 5)        
                # Fit PCA/FCCA
                cross_covs_rev = [np.linalg.inv(cross_covs[0]) @ c.T @ np.linalg.inv(cross_covs[0]) for c in cross_covs]
                cross_covs = torch.tensor(cross_covs)
                cross_covs_rev = torch.tensor(cross_covs_rev)
                e, Upca = np.linalg.eig(cross_covs[0])
                eigorder = np.argsort(e)[::-1]
                pca_coef = Upca[:, eigorder][:, 0:d]
                lqgmodel = LQGCA(d=d, T=T, rng_or_seed=int(inner_rep))
                lqgmodel.cross_covs = cross_covs
                lqgmodel.cross_covs_rev = cross_covs_rev
                coef_, score = lqgmodel._fit_projection()

                rd = {}
                rd['rep'] = rep
                rd['inner_rep'] = inner_rep
                rd['R'] = r
                rd['pca_coef'] = pca_coef
                rd['fcca_coef'] = coef_
                rd['fcca_score'] = score
                rd['pca_eig'] = e[eigorder][0:d]
                rd['dim'] = d
                rd['T'] = T
                rd['nn'] = nn
            except:
                cross_covs = np.nan
                rd = {} 
      
>>>>>>> Stashed changes
        print('Rank %d Completed task %d/%d in %f' % (comm.rank, i + 1, len(tasks), time.time() - t0))

        # save to file (append)
        with open('%s/rank%d.pkl' % (savepath, comm.rank), 'ab') as f:
            f.write(pickle.dumps(task))
            f.write(pickle.dumps(seq_idxs))
            f.write(pickle.dumps(A))
            f.write(pickle.dumps(nn))
            f.write(pickle.dumps(cross_covs))
            f.write(pickle.dumps(rd))
