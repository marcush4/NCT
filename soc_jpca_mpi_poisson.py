import itertools
import os
import time
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
R = np.linspace(0.75, 10, 25)
<<<<<<< Updated upstream
R = np.concatenate([R, np.array([11, 12, 13, 14, 15])])
=======
#R = np.concatenate([R, np.array([11, 12, 13, 14, 15])])
#R = np.array([11, 12, 13, 14, 15])
>>>>>>> Stashed changes

def gen_activity(tau, W, activ_func, sigma, T, h):

    # f
    def f_(x, t):
        return 1/tau * (-1 * np.eye(W.shape[0]) @ x + W @ activ_func(x))

    # G: linear i.i.d noise with sigma
    def g_(x, t):
        return sigma * np.eye(W.shape[0])

    # Generate random initial condition and then integrate over the desired time period
    tspace = np.linspace(0, T, int(T/h))
    
    x0 = np.random.normal(size=(W.shape[0],))

    return  sdeint.itoSRI2(f_, g_, x0, tspace)    

def calc_cross_covs(tau, W, sigma, T, h):
    A = 1/tau * (-1 * np.eye(W.shape[0]) @ x + W @ x)
    B = sigma * np.eye(A.shape[0])
    P = scipy.linalg.solve_continuous_lyapunov()
    


def sample_activity(tau, W, sigma, T, h):


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
    activitypath = sys.argv[1]
    savepath = sys.argv[2]

    dt = 1
    d = 6

    tasks = list(itertools.product(np.arange(reps), np.arange(inner_reps), R))
    tasks = np.array_split(tasks, comm.size)[comm.rank]

    #phi = np.zeros(len(tasks))
    #scores = np.zeros(len(tasks))
    #nn = np.zeros((len(tasks), 2))
    #jpca_eig = np.zeros((len(tasks), 2))

    # #Alist = []

    # First load generated A matrics
    if comm.rank == 0:
<<<<<<< Updated upstream
        with open('soc_Alist.pkl', 'rb') as f:
=======
        with open('Alist_lambctrl.pkl', 'rb') as f:
>>>>>>> Stashed changes
            Alist = pickle.load(f)
    else:
        Alist = None
    
    Alist = comm.bcast(Alist)

    print(len(tasks))

    for i, task in enumerate(tasks):
        t0 = time.time()
        rep, inner_rep, r = task
        rep = int(rep)
        inner_rep = int(inner_rep)
        ridx = list(R).index(r)
        A = Alist[rep][ridx]
        nn = np.linalg.norm(A @ A.T - A.T @ A)
        #Alist.append(A)
        # Simulate from the model
        # x = gen_activity(1, A, lambda x: x, 1, 1000, 1e-1)   
        #print('activity gen')

        # Load the activity for the task
<<<<<<< Updated upstream
        with open('%s/x%d_%d_%d.pkl' % (activitypath, rep, r, inner_rep), 'rb') as f:
            x = pickle.load(f)

        # Sample spikes and calculate cross-covariance matrices
        ccm1 = calc_cross_cov_mats_from_data(x, 5)        
        boxcox = 0.5
        spike_rates_trials = []
        #spike_rate_ = np.exp(x)
        #spike_counts = np.random.poisson(np.tile(spike_rate_[np.newaxis, :], (500, 1)))
        # spike_rates_trials = (np.power(spike_counts, boxcox) - 1)/boxcox
        for _ in tqdm(range(100)):
            spike_counts = np.random.poisson(np.exp(x))
            spike_rates = np.array([(np.power(spike_count, boxcox) - 1)/boxcox 
                                    for spike_count in spike_counts])
            #spike_rates = np.array([log0(spike_count) 
            #                           for spike_count in spike_counts])
            spike_rates = np.array([scipy.stats.boxcox(spike_count, boxcox) for spike_count in spike_counts])
            spike_rates_trials.append(spike_rates)
        spike_rates_trials = np.array(spike_rates_trials)
        print('Calculating ccm')
        ccm2 = calc_cross_cov_mats_from_data(spike_rates_trials, 5)        

        print('Rank %d Completed task %d/%d in %f' % (comm.rank, i + 1, len(tasks), time.time() - t0))

        # save to file (append)
        with open('%s/rank%d.pkl' % (savepath, comm.rank), 'ab') as f:
            f.write(pickle.dumps(task))
            f.write(pickle.dumps(A))
            f.write(pickle.dumps(nn))
            f.write(pickle.dumps(ccm1))
            f.write(pickle.dumps(ccm2))
=======
        fname = '%s/x_%d_%d_%d.pkl' % (activitypath, ridx, rep, inner_rep)
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                x = pickle.load(f)


            # Sample spikes and calculate cross-covariance matrices
            ccm1 = calc_cross_cov_mats_from_data(x, 5)        
            boxcox = 0.5
            spike_rates_trials = []
            #spike_rate_ = np.exp(x)
            #spike_counts = np.random.poisson(np.tile(spike_rate_[np.newaxis, :], (500, 1)))
            # spike_rates_trials = (np.power(spike_counts, boxcox) - 1)/boxcox
            for _ in tqdm(range(100)):
                spike_counts = np.random.poisson(np.exp(x))
                spike_rates = np.array([(np.power(spike_count, boxcox) - 1)/boxcox 
                                        for spike_count in spike_counts])
                #spike_rates = np.array([log0(spike_count) 
                #                           for spike_count in spike_counts])
                spike_rates = np.array([scipy.stats.boxcox(spike_count, boxcox) for spike_count in spike_counts])
                spike_rates_trials.append(spike_rates)
            spike_rates_trials = np.array(spike_rates_trials)
            print('Calculating ccm')
            ccm2 = calc_cross_cov_mats_from_data(spike_rates_trials, 5)        

            print('Rank %d Completed task %d/%d in %f' % (comm.rank, i + 1, len(tasks), time.time() - t0))

            # save to file (append)
            with open('%s/rank%d.pkl' % (savepath, comm.rank), 'ab') as f:
                f.write(pickle.dumps(task))
                f.write(pickle.dumps(A))
                f.write(pickle.dumps(nn))
                f.write(pickle.dumps(ccm1))
                f.write(pickle.dumps(ccm2))
        else:
            print('activity file not found, skipping')
>>>>>>> Stashed changes
