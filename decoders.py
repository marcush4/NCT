import numpy as np
import scipy
from scipy import signal
import pdb
from copy import deepcopy

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, hamming_loss, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from pykalman import KalmanFilter
import PSID

# Stuff for reduced rank logistic regression
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#from pyuoi.linear_model.var import VAR
#from pyuoi.utils import log_likelihood_glm
#from dca.dca import DynamicalComponentsAnalysis as DCA
from dca.cov_util import (calc_cross_cov_mats_from_data, 
                          calc_pi_from_cross_cov_mats, form_lag_matrix)

def decimate_(X, q):

    Xdecimated = []
    for i in range(X.shape[1]):
        Xdecimated.append(signal.decimate(X[:, i], q))

    return np.array(Xdecimated).T

# If X has trial structure, need to seperately normalize each trial
def standardize(X):

    scaler = StandardScaler()

    if type(X) == list:
        Xstd = [scaler.fit_transform(x) for x in X] 
    elif np.ndim(X) == 3:
        Xstd = np.array([scaler.fit_transform(X[idx, ...]) 
                         for idx in range(X.shape[0])])
    else:
        Xstd = scaler.fit_transform(X)

    return Xstd

# Turn position into velocity and acceleration with finite differences
def expand_state_space(Z, X, include_vel=True, include_acc=True):

    concat_state_space = []
    for i, z in enumerate(Z):
        if include_vel and include_acc:
            pos = z[2:, :]
            vel = np.diff(z, 1, axis=0)[1:, :]
            acc = np.diff(z, 2, axis=0)

            # Trim off 2 samples from the neural data to match lengths
            X[i] = X[i][2:, :]

            concat_state_space.append(np.concatenate((pos, vel, acc), axis=-1))
        elif include_vel:
            pos = z[1:, :]
            vel = np.diff(z, 1, axis=0)
            # Trim off only one sample in this case
            X[i] = X[i][1:, :]
            concat_state_space.append(np.concatenate((pos, vel), axis=-1))
        else:
            concat_state_space.append(z)

    return concat_state_space, X
    
def KF(X, Z):

    # Assemble kinematic state variable (6D)
    # Chop off the first 2 points for equal length vectors
    pos = Z[2:, :]
    vel = np.diff(Z, 1, axis=0, )[1:, :]
    acc = np.diff(Z, 2, axis=0, )

    z = np.hstack([pos, vel, acc])

    # Trim neural data accordingly
    x = X[2:, :]

    # Kinematic mean and variance (same-time)
    mu0 = np.mean(z, axis=0)
    Sigma0 = np.cov(z.T)

    # Kinematic state transition matrices
    linregressor = LinearRegression(normalize=True, fit_intercept=True)
    varmodel = VAR(estimator='ols', fit_intercept=True, order=1, 
                   self_regress=True)
    varmodel.fit(z)

    A = np.squeeze(varmodel.coef_)
    az = varmodel.intercept_

    # Get the residual covariance
    zpred, z_ = varmodel.predict(z)

    epsilon = z_ - zpred
    Sigmaz = np.cov(epsilon.T)

    # Predict the neural data from the kinematic data
    try:
        linregressor.fit(z, x)
    except:
        pdb.set_trace()
    Cxz = linregressor.coef_
    cxz = linregressor.intercept_

    # Can try to do poisson regression here

    yypred = linregressor.predict(z)
    epsilon = x - yypred
    Sigmaxz = np.cov(epsilon.T)

    # Instantiate a Kalman filter using these parameters
    kf = KalmanFilter(transition_matrices = A, observation_matrices = Cxz,
                      transition_covariance = Sigmaz, observation_covariance = Sigmaxz,
                      transition_offsets = az, observation_offsets = cxz,
                      initial_state_mean = mu0, initial_state_covariance=Sigma0)

    return kf

def kf_decoder(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window=1):

    Ztrain = scaler.fit_transform(Ztrain)
    Xtrain = scaler.fit_transform(Xtrain)

    Xtest = scaler.fit_transform(Xtest)
    Ztest = scaler.fit_transform(Ztest)

    # Apply train lag
    if trainlag > 0:
        Xtrain = Xtrain[:-trainlag, :]
        Ztrain = Ztrain[trainlag:, :]

    if testlag > 0:
        # Apply test lag
        Xtest = Xtest[:-testlag, :]
        Ztest = Ztest[testlag:, :]

    kf = KF(Xtrain, Ztrain)

    state_estimates, _ = kf.filter(Xtest)

    pos_estimates = state_estimates[:, 0:2]
    vel_estimates = state_estimates[:, 2:4]
    acc_estimates = state_estimates[:, 2:, ]

    pos_true = Ztest[2:, :]
    vel_true = np.diff(Ztest, 1, axis=0, )[1:, :]
    acc_true = np.diff(Ztest, 2, axis=0, )

    kf_r2_pos = r2_score(pos_true, pos_estimates)
    kf_r2_vel = r2_score(vel_true, vel_estimates)
    kf_r2_acc = r2_score(acc_true, acc_estimates)

    return kf_r2_pos, kf_r2_vel, kf_r2_acc, kf

def lr_preprocess(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window, include_velocity, include_acc):

    # If no trial structure is present, convert to a list for easy coding
    if np.ndim(Xtrain) == 2:
        Xtrain = [Xtrain]
        Xtest = [Xtest]

        Ztrain = [Ztrain]
        Ztest = [Ztest]

    Ztrain = standardize(Ztrain)
    Xtrain = standardize(Xtrain)

    Ztest = standardize(Ztest)
    Xtest = standardize(Xtest)

    # Apply train lag
    if trainlag > 0:
        Xtrain = [x[:-trainlag, :] for x in Xtrain]
        Ztrain = [z[trainlag:, :] for z in Ztrain]
    elif trainlag < 0:
        Xtrain = [x[-trainlag:, :] for x in Xtrain]
        Ztrain = [z[:trainlag, :] for z in Ztrain]


    # Apply test lag
    if testlag > 0:
        Xtest = [x[:-trainlag, :] for x in Xtest]
        Ztest = [z[trainlag:, :] for z in Ztest]
    elif testlag < 0:
        Xtest = [x[-trainlag:, :] for x in Xtest]
        Ztest = [z[:trainlag, :] for z in Ztest]

    # Apply decoding window
    Xtrain = [form_lag_matrix(x, decoding_window) for x in Xtrain]
    Xtest = [form_lag_matrix(x, decoding_window) for x in Xtest]

    Ztrain = [z[decoding_window//2:, :] for z in Ztrain]
    Ztrain = [z[:x.shape[0], :] for z, x in zip(Ztrain, Xtrain)]

    Ztest = [z[decoding_window//2:, :] for z in Ztest]
    Ztest = [z[:x.shape[0], :] for z, x in zip(Ztest, Xtest)]

    # Expand state space to include velocity and acceleration
    if np.any([include_velocity, include_acc]):
        Ztrain, Xtrain = expand_state_space(Ztrain, Xtrain, include_velocity, include_acc)
        Ztest, Xtest = expand_state_space(Ztest, Xtest, include_velocity, include_acc)

    # Flatten trial structure as regression will not care about it
    Xtrain = np.concatenate(Xtrain)
    Xtest = np.concatenate(Xtest)
    Ztrain = np.concatenate(Ztrain)
    Ztest = np.concatenate(Ztest)

    return Xtest, Xtrain, Ztest, Ztrain

# Sticking with consistent nomenclature, Z is the behavioral data and X is the neural data
def lr_encoder(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window=1, include_velocity=True, include_acc=False):

    # By default, we look only at pos and vel
    Xtest, Xtrain, Ztest, Ztrain = lr_preprocess(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window, include_velocity, include_acc)

    # Apply the decoding window to the behavioral data
    # Ztrain, _ = form_lag_matrix(Ztrain, decoding_window)
    # Ztest, _ = form_lag_matrix(Ztest, decoding_window)

    # Xtrain = Xtrain[decoding_window//2:, :]
    # Xtest = Xtest[:Ztest.shape[1], :]

    encodingregressor = LinearRegression(fit_intercept=True)

    # Throw away acceleration
    # Ztest = Ztest[:, 0:4]
    # Ztrain = Ztrain[:, 0:4]

    encodingregressor.fit(Ztrain, Xtrain)
    Xpred = encodingregressor.predict(Ztest)

    r2 = r2_score(Xtest, Xpred)
    return r2, encodingregressor

def svm_decoder(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window=1, include_velocity=True, include_acc=True):

    # Do not fit if the sample size is too large
    if Xtrain.shape[0] > 20000:
        return np.nan, np.nan, np.nan, None, np.nan, np.nan, np.nan

    # Only fit every other dimension
    # Load arg files and check if dimensionality is *every other*
    dimvals = np.arange(1, 31, 2)
    if Xtrain.shape[-1] not in dimvals:
        return np.nan, np.nan, np.nan, None, np.nan, np.nan, np.nan

    behavior_dim = Ztrain[0].shape[-1]
    Xtest, Xtrain, Ztest, Ztrain = lr_preprocess(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window, include_velocity, include_acc)
    svr = SVR()
    decodingregressor = MultiOutputRegressor(svr)
    decodingregressor.fit(Xtrain, Ztrain)
    Zpred = decodingregressor.predict(Xtest)
    # Calculate log likelihood of the training fit
    if include_velocity and include_acc:
        lr_r2_pos = r2_score(Ztest[..., 0:behavior_dim], Zpred[..., 0:behavior_dim])
        lr_r2_vel = r2_score(Ztest[..., behavior_dim:2*behavior_dim], Zpred[..., behavior_dim:2*behavior_dim])
        lr_r2_acc = r2_score(Ztest[..., 2*behavior_dim:], Zpred[..., 2*behavior_dim:])

        return lr_r2_pos, lr_r2_vel, lr_r2_acc, decodingregressor, np.nan, np.nan, np.nan
    elif include_velocity:
        lr_r2_pos = r2_score(Ztest[..., 0:behavior_dim], Zpred[..., 0:behavior_dim])
        lr_r2_vel = r2_score(Ztest[..., behavior_dim:2*behavior_dim], Zpred[..., behavior_dim:2*behavior_dim])
        return lr_r2_pos, lr_r2_vel, decodingregressor, np.nan, np.nan
    else:
        lr_r2_pos = r2_score(Ztest[..., 0:behavior_dim], Zpred[..., 0:behavior_dim])
        lr_r2_pos = r2_score(Ztest, Zpred)
        return lr_r2_pos, decodingregressor, np.nan

def psid_decoder(Xtest, Xtrain, Ztest, Ztrain, lag, include_velocity=True, include_acc=True, state_dim=None):

    behavior_dim = Ztrain[0].shape[-1]
    Xtest, Xtrain, Ztest, Ztrain = lr_preprocess(Xtest, Xtrain, Ztest, Ztrain, 0, 0, 1, include_velocity, include_acc)
    if state_dim is None:
        try:
            state_dim = Xtrain.shape[-1]
        except:
            state_dim = Xtrain[0].shape[-1]
    elif np.isscalar(state_dim):
        # Was a single state dimension passed in?
        try:
            idsys = PSID.PSID(Xtrain, Ztrain, nx=state_dim, n1=state_dim, i=lag)
            Zpred, _, _ = idsys.predict(Xtest)

            lr_r2_pos = r2_score(Ztest[..., 0:behavior_dim], Zpred[..., 0:behavior_dim])
            lr_r2_vel = r2_score(Ztest[..., behavior_dim:2*behavior_dim], Zpred[..., behavior_dim:2*behavior_dim])
            lr_r2_acc = r2_score(Ztest[..., 2*behavior_dim:], Zpred[..., 2*behavior_dim:])
        except:
            # SVD likely has failed to converge
            print('SVD failed to converge!')
            lr_r2_pos = np.nan
            lr_r2_vel = np.nan
            lr_r2_acc = np.nan

        # r2 = evalPrediction(Ztest, zpred, 'R2')
        return lr_r2_pos, lr_r2_vel, lr_r2_acc, None, None, None, None 
    else:
        # An array of state dimensions was passed in that we should loop over
        r2_pos = np.zeros(len(state_dim))
        r2_vel = np.zeros(len(state_dim))
        r2_acc = np.zeros(len(state_dim))

        for i, sdim in enumerate(state_dim):
            idsys = PSID.PSID(Xtrain, Ztrain, nx=sdim, n1=sdim, i=lag)
            Zpred, _, _ = idsys.predict(Xtest)

            lr_r2_pos = r2_score(Ztest[..., 0:behavior_dim], Zpred[..., 0:behavior_dim])
            lr_r2_vel = r2_score(Ztest[..., behavior_dim:2*behavior_dim], Zpred[..., behavior_dim:2*behavior_dim])
            lr_r2_acc = r2_score(Ztest[..., 2*behavior_dim:], Zpred[..., 2*behavior_dim:])
            # except:
            #     # SVD likely has failed to converge
            #     # print('SVD failed to converge!')
            #     lr_r2_pos = np.nan
            #     lr_r2_vel = np.nan
            #     lr_r2_acc = np.nan
            r2_pos[i] = lr_r2_pos
            r2_vel[i] = lr_r2_vel
            r2_acc[i] = lr_r2_acc
        return r2_pos, r2_vel, r2_acc, None, None, None, None 

def rrlr_decoder(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, ranks, decoding_window=1, include_velocity=True, include_acc=True):
    # Don't have the trailized version currently
    if isinstance(Xtrain, list) or np.ndim(Xtrain) == 3:
        raise NotImplementedError

    behavior_dim = Ztrain[0].shape[-1]
    Xtest, Xtrain, Ztest, Ztrain = lr_preprocess(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window, include_velocity, include_acc)
    # Following the exposition in 
    # Sparse Reduced-Rank Regression for Simultaneous Dimension Reduction and Variable Selection

    # Form the matrices Sxx and Sxy
    n = Xtrain.shape[0]
    Sxx = 1/n * Xtrain.T @ Xtrain
    Sxxinv = np.linalg.pinv(Sxx)
    Sxy = 1/n * Xtrain.T @ Ztrain
    eig, V = np.linalg.eig(Sxy.T @ Sxxinv @ Sxy)

    eigorder = np.argsort(eig)[::-1]
    V = V[:, eigorder]
    
    r2_pos = np.zeros(ranks.size)
    r2_vel = np.zeros(ranks.size)
    r2_acc = np.zeros(ranks.size)

    for i, rank in enumerate(ranks):
        Vr = V[:, 0:rank]
        coef = Sxxinv @ Sxy @ Vr @ Vr.T

        # Predict
        Zpred = Xtest @ coef
        r2_pos[i] = r2_score(Ztest[:, 0:behavior_dim], Zpred[:, 0:behavior_dim])

        if include_velocity:            
            r2_vel[i] = r2_score(Ztest[..., behavior_dim:2*behavior_dim], 
                                 Zpred[..., behavior_dim:2*behavior_dim])
        if include_acc:
            r2_acc[i] = r2_score(Ztest[..., 2*behavior_dim:], 
                                 Zpred[..., 2*behavior_dim:])
            
    return r2_pos, r2_vel, r2_acc, None, None, None, None        

def lr_decoder(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window=1, include_velocity=True, include_acc=True):

    behavior_dim = Ztrain[0].shape[-1]
    Xtest, Xtrain, Ztest, Ztrain = lr_preprocess(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window, include_velocity, include_acc)
    decodingregressor = LinearRegression(fit_intercept=True)
    decodingregressor.fit(Xtrain, Ztrain)
    Zpred = decodingregressor.predict(Xtest)

    # Calculate log likelihood of the training fit
    Zpred_train = decodingregressor.predict(Xtrain)
    if include_velocity and include_acc:
        #logll_pos = log_likelihood_glm('normal', Ztrain[..., 0:behavior_dim], Zpred_train[..., 0:behavior_dim])
        #logll_vel = log_likelihood_glm('normal', Ztrain[..., behavior_dim:2*behavior_dim], Zpred_train[..., behavior_dim:2*behavior_dim])
        #logll_acc = log_likelihood_glm('normal', Ztrain[..., 2*behavior_dim:], Zpred_train[..., 2*behavior_dim:])

        logll_pos = np.nan
        logll_vel = np.nan
        logll_acc = np.nan
    
        lr_r2_pos = r2_score(Ztest[..., 0:behavior_dim], Zpred[..., 0:behavior_dim])
        lr_r2_vel = r2_score(Ztest[..., behavior_dim:2*behavior_dim], Zpred[..., behavior_dim:2*behavior_dim])
        lr_r2_acc = r2_score(Ztest[..., 2*behavior_dim:], Zpred[..., 2*behavior_dim:])

        return lr_r2_pos, lr_r2_vel, lr_r2_acc, decodingregressor, logll_pos, logll_vel, logll_acc
    elif include_velocity:

        logll_pos = log_likelihood_glm('normal', Ztrain[..., 0:behavior_dim], Zpred_train[..., 0:behavior_dim])
        logll_vel = log_likelihood_glm('normal', Ztrain[..., behavior_dim:2*behavior_dim], Zpred_train[..., behavior_dim:2*behavior_dim])


        lr_r2_pos = r2_score(Ztest[..., 0:behavior_dim], Zpred[..., 0:behavior_dim])
        lr_r2_vel = r2_score(Ztest[..., behavior_dim:2*behavior_dim], Zpred[..., behavior_dim:2*behavior_dim])


        return lr_r2_pos, lr_r2_vel, decodingregressor, logll_pos, logll_vel
    else:

        logll_pos = log_likelihood_glm('normal', Ztrain[..., 0:behavior_dim], Zpred_train[..., 0:behavior_dim])
        logll_vel = log_likelihood_glm('normal', Ztrain[..., behavior_dim:2*behavior_dim], Zpred_train[..., behavior_dim:2*behavior_dim])

        lr_r2_pos = r2_score(Ztest[..., 0:behavior_dim], Zpred[..., 0:behavior_dim])
        lr_r2_pos = r2_score(Ztest, Zpred)
        return lr_r2_pos, decodingregressor, logll_pos

def _draw_bootstrap_sample(rng, X, y):
    sample_indices = np.arange(X.shape[0])
    bootstrap_indices = rng.choice(
        sample_indices, size=sample_indices.shape[0], replace=True
    )
    return X[bootstrap_indices], y[bootstrap_indices]


def lr_bias_variance(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window=1, n_boots=200, random_seed=None):

    if random_seed is None:
        rand = np.random
    else:
        rand = np.random.RandomState(random_seed)

    # To bootstrap, we need to preprocess and flatten the data
    Xtest, Xtrain, Ztest, Ztrain = lr_preprocess(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window, include_velocity=True, include_acc=True)
    # Run lr_decoder over bootstrapped samples of xtrain and xtest. Use this to calculate bias and variance of the estimator
    zpred_boot = []
    for k in range(n_boots):
        xboot, zboot = _draw_bootstrap_sample(rand, Xtrain, Ztrain)
        decodingregressor = LinearRegression(fit_intercept=True)
        decodingregressor.fit(xboot, zboot)
        zpred = decodingregressor.predict(Xtest)
        zpred_boot.append(zpred)

    zpred_boot = np.array(zpred_boot)

    assert(np.allclose((zpred_boot - Ztest).shape, zpred_boot.shape))

    # Bias/Variance/MSE
    mse = np.mean(np.mean(np.power(zpred_boot - Ztest, 2), axis=1), axis=0)
    Ezpred = np.mean(zpred_boot, axis=0)
    bias = np.sum((Ezpred - Ztest)**2, axis=0)/Ztest.shape[0]
    var = np.mean(np.mean(np.power(zpred_boot - Ezpred, 2), axis=1), axis=0)
    return mse, bias, var

#*** Assumes that Z already has dimension 6**** (hence includevelocity/acc is set to FALSE)#
# This also means that the transition times and pkassign indices correspond to Z having been passed through expnad state space #
def lr_bv_windowed(X, Z, lag, train_windows, test_windows, transition_times, train_idxs, test_idxs, pkassign=None, apply_pk_to_train=False, 
                   decoding_window=1, n_boots=200, random_seed=None, offsets=None, norm=False):

    if random_seed is None:
        rand = np.random
    else:
        rand = np.random.RandomState(random_seed)

    win_min = train_windows[0][0]

    if win_min >= 0:
        win_min = 0

    # Filter out by transitions that lie within the train idxs, and stay clear of the start and end
    tt_train = [(t, idx) for idx, t in enumerate(transition_times) 
                if idx in train_idxs and t[0] > (lag + np.abs(win_min)) and t[1] < (Z.shape[0] - lag - np.abs(win_min))]
    # Re-assign train idxs removing those reaches that were outside the start/end region
    train_idxs = [x[1] for x in tt_train]
    tt_train = [x[0] for x in tt_train]

    if offsets is not None:
        offsets_train = offsets[train_idxs]
    else:
        offsets_train = None

    # Get trialized, windowed data
    if pkassign is not None and apply_pk_to_train:
        assert(np.all([s.size == np.arange(t[0], t[1]).size for (s, t) in zip(pkassign[train_idxs], tt_train)]))
        subset_selection = [np.argwhere(np.array(s) == 0).squeeze() for s in pkassign[train_idxs]]

        Xtrain, Ztrain, _, _ = apply_window(X, Z, lag, train_windows, tt_train, decoding_window, False, False, subset_selection, offsets=offsets_train)
    else:
        Xtrain, Ztrain, _, _ = apply_window(X, Z, lag, train_windows, tt_train, decoding_window, False, False, offsets=offsets_train)

    # Filter out by transitions that lie within the test idxs, and stay clear of the start and end
    tt_test = [(t, idx) for idx, t in enumerate(transition_times) 
                if idx in test_idxs and t[0] > (lag + np.abs(win_min)) and t[1] < (Z.shape[0] - lag - np.abs(win_min))]
    # Re-assign test idxs removing those reaches that were outside the start/end region
    test_idxs = [x[1] for x in tt_test]
    tt_test = [x[0] for x in tt_test]
    if offsets is not None:
        offsets_test = offsets[test_idxs]
    else:
        offsets_test = None

    if pkassign is not None:
        assert(np.all([s.size == np.arange(t[0], t[1]).size for (s, t) in zip(pkassign[test_idxs], tt_test)]))
        subset_selection = [np.argwhere(np.array(s) != 0).squeeze() for s in pkassign[test_idxs]]
        Xtest, Ztest, _, _ = apply_window(X, Z, lag, test_windows, tt_test, decoding_window, False, False, subset_selection, offsets=offsets_test)
    else:
        Xtest, Ztest, _, _ = apply_window(X, Z, lag, test_windows, tt_test, decoding_window, False, False, offsets=offsets_test)

    num_test_reaches = len(Xtest)
    # verify dimensionalities
    if len(Xtrain) > 0:
        Xtrain = np.concatenate(Xtrain)
        Ztrain = np.concatenate(Ztrain)
    else:
        return np.nan, np.nan, np.nan, num_test_reaches

    if len(Xtest) > 0:
        Xtest = np.concatenate(Xtest)
        Ztest = np.concatenate(Ztest)

        Xtrain = StandardScaler().fit_transform(Xtrain)
        #Ztrain = StandardScaler().fit_transform(Ztrain)
        Xtest = StandardScaler().fit_transform(Xtest)
        #Ztest = StandardScaler().fit_transform(Ztest)

        # Run lr_decoder over bootstrapped samples of xtrain and xtest. Use this to calculate bias and variance of the estimator
        zpred_boot = []
        for k in range(n_boots):

            xboot, zboot = _draw_bootstrap_sample(rand, Xtrain, Ztrain)
            decodingregressor = LinearRegression(fit_intercept=True)
            decodingregressor.fit(xboot, zboot)
            zpred = decodingregressor.predict(Xtest)
            zpred_boot.append(zpred)

        zpred_boot = np.array(zpred_boot)

        assert(np.allclose((zpred_boot - Ztest).shape, zpred_boot.shape))

        # Bias/Variance/MSE
        mse = np.mean(np.mean(np.power(zpred_boot - Ztest, 2), axis=1), axis=0)
        
        Ezpred = np.mean(zpred_boot, axis=0)
        bias = np.sum((Ezpred - Ztest)**2, axis=0)/Ztest.shape[0]
        var = np.mean(np.mean(np.power(zpred_boot - Ezpred, 2), axis=1), axis=0)
        if norm:
            norm_ = np.mean(np.power(Ztest, 2), axis=0)
            return np.divide(mse, norm_), np.divide(bias, norm_), np.divide(var, norm_), num_test_reaches
        else:
            return mse, bias, var, num_test_reaches
    else:
        return np.nan, np.nan, np.nan, num_test_reaches


def apply_window(X, Z, lag, window, transition_times, decoding_window, include_velocity, include_acc, 
                 subset_selection=None, offsets=None, enforce_full_indices=False):
    # Update 12/15: We allow for multiple windows for each transition time, so we can train the decoder across pooled sections
    # of the reach

    # subset_selection: set of indices of the same length as transition_times that indicate whether a subset of the transition
    # is to be included. This is used when we enforce peak membership in decoding.

    # Apply decoding window
    X = form_lag_matrix(X, decoding_window)
    Z = Z[decoding_window//2:, :]
    Z = Z[:X.shape[0], :]

    # Expand state space to include velocity and acceleration
    if np.any([include_velocity, include_acc]):
        Z, X = expand_state_space([Z], [X], include_velocity, include_acc)

        # Flatten list structure imposed by expand_state_space
        Z = Z[0]
        X= X[0]

    # This *also* requires shifting the transition times, as behavior will have been affected
    # There is a shift due to the formation of the lag matrix *and* the expansion of the state sapce, 
    # which will cut off the first 2 sample points, if include_acc is set to true

    if decoding_window > 1:
        transition_times = [(t[0] - decoding_window//2, t[1] - decoding_window//2) for t in transition_times]        
    try:
        assert(X.shape[0] == Z.shape[0])
    except:
        pdb.set_trace()

    if include_acc:
        transition_times = [(t[0] - 2, t[1] - 2) for t in transition_times]
    elif include_velocity:
        transition_times = [(t[0] - 1, t[1] - 1) for t in transition_times]

    # We assume that the subset indices have already been shifted (this is the case in biasvariance_vst)

    # Segment the time series with respect to the transition times (including lag)
    xx = []
    zz = []

    valid_idxs = []
    # Which reaches had no truncation due to start of the next reach?
    full_idxs = []

    # If given a single window, duplicate it across all transition times
    if len(window) != len(transition_times):
        window = [window for _ in range(len(transition_times))]

    # If no offsets provided, let it be 0 for all transition times
    if offsets is None:
        offsets = np.zeros(len(transition_times))

    for i, (t0, t1) in enumerate(transition_times):
        for j, win in enumerate(window[i]):
            window_indices = np.arange(t0 + win[0] + offsets[i], t0 + win[1] + offsets[i])
            if subset_selection is not None:
                # Select only indices that do not belong to the first velocity peak
                subset_indices = np.arange(t0, t1)[subset_selection[i]]
                window_indices = np.intersect1d(window_indices, subset_indices)

            # No matter what, we should remove segments that overlap with the next transition
            if i < len(transition_times) - 1:
                l1 = len(window_indices)
                window_indices = window_indices[window_indices < transition_times[i + 1][0]]
                if len(window_indices) == l1:
                    full_idxs.append(i)
            else:
                # Or else make sure that we don't exceed the length of the time series
                window_indices = window_indices[window_indices < Z.shape[0]]

            window_indices = window_indices.astype(int)
            zz_ = Z[window_indices]

            # Shift x indices by lag
            window_indices -= lag
            xx_ = X[window_indices]

            if len(xx_) > 0:
                assert(xx_.shape[0] == zz_.shape[0])
                if enforce_full_indices:
                    if i in full_idxs:
                        xx.append(xx_)
                        zz.append(zz_)
                else:
                    xx.append(xx_)
                    zz.append(zz_)
                valid_idxs.append(i)
                # try:
                #     assert(full_idxs[-1] in valid_idxs or i == len(transition_times) - 1)
                # except:
                #     pdb.set_trace()
    return xx, zz, valid_idxs, full_idxs
    
def lr_decode_windowed(X, Z, lag, train_windows, test_windows, transition_times, train_idxs, test_idxs=None, 
                       decoding_window=1, include_velocity=True, include_acc=True, 
                       pkassign=None, apply_pk_to_train=False, offsets=None):

    behavior_dim = Z.shape[-1]

    # We have been given a list (of list) of windows for each transition
    win_min = train_windows[0][0]

    if win_min >= 0:
        win_min = 0

    # Filter out by transitions that lie within the train idxs, and stay clear of the start and end
    tt_train = [(t, idx) for idx, t in enumerate(transition_times) 
                if idx in train_idxs and t[0] > (lag + np.abs(win_min)) and t[1] < (Z.shape[0] - lag - np.abs(win_min))]
    # Re-assign train idxs removing those reaches that were outside the start/end region
    train_idxs = [x[1] for x in tt_train]
    tt_train = [x[0] for x in tt_train]

    if offsets is not None:
        offsets_train = offsets[train_idxs]
    else:
        offsets_train = None

    if apply_pk_to_train:
        # Train on the first velocity peak only
        assert(np.all([s.size == np.arange(t[0], t[1]).size for (s, t) in zip(pkassign[train_idxs], tt_train)]))
        subset_selection = [np.argwhere(np.array(s) == 0).squeeze() for s in pkassign[train_idxs]]
        Xtrain, Ztrain, vi1, fi1 = apply_window(X, Z, lag, train_windows, tt_train, decoding_window, include_velocity, include_acc, subset_selection, offsets=offsets_train,
                                                enforce_full_indices=True)

    else:
        Xtrain, Ztrain, vi1, fi1 = apply_window(X, Z, lag, train_windows, tt_train, decoding_window, include_velocity, include_acc, offsets=offsets_train,
                                      enforce_full_indices=True)

    if Xtrain is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None, 0
    else:
        n = len(Xtrain)

    if test_idxs is not None:
        # Filter out by transitions that lie within the test idxs, and stay clear of the start and end
        tt_test = [(t, idx) for idx, t in enumerate(transition_times) 
                   if idx in test_idxs and t[0] > (lag + np.abs(win_min)) and t[1] < (Z.shape[0] - lag - np.abs(win_min))]
        # Re-assign test idxs removing those reaches that were outside the start/end region
        test_idxs = [x[1] for x in tt_test]
        tt_test = [x[0] for x in tt_test]


        if offsets is not None:
            offsets_test = offsets[test_idxs]
        else:
            offsets_test = None

        if pkassign is not None:
            assert(np.all([s.size == np.arange(t[0], t[1]).size for (s, t) in zip(pkassign[test_idxs], tt_test)]))
            subset_selection = [np.argwhere(np.array(s) != 0).squeeze() for s in pkassign[test_idxs]]
            Xtest, Ztest, _, fi2 = apply_window(X, Z, lag, test_windows, tt_test, decoding_window, include_velocity, include_acc, subset_selection, offsets=offsets_test, 
                                               enforce_full_indices=True)        
        else:
            Xtest, Ztest, _, fi2 = apply_window(X, Z, lag, test_windows, tt_test, decoding_window, include_velocity, include_acc, offsets=offsets_test,
                                               enforce_full_indices=True)

    else:
        Xtest = None
        Ztest = None

    # Standardize
    # X = StandardScaler().fit_transform(X)
    # Z = StandardScaler().fit_transform(Z)
    decodingregressor = LinearRegression(fit_intercept=True)

    # Fit and score
    if len(Xtrain) == 0:
        return tuple([np.nan] * 9) + (0,)
    decodingregressor.fit(np.concatenate(Xtrain), np.concatenate(Ztrain))
    Zpred = decodingregressor.predict(np.concatenate(Xtrain))

    # Re-segment Zpred
    idx = 0
    Zpred_segmented = []
    for i, z in enumerate(Ztrain):
        Zpred_segmented.append(Zpred[idx:idx+z.shape[0]])
        idx += z.shape[0]

    assert(np.all([z1.shape[0] == z2.shape[0] for (z1, z2) in zip(Zpred_segmented, Ztrain)]))
    #Ztrain = np.concatenate(Ztrain)
    if Xtest is not None:
        if len(Xtest) > 0:
            num_test_reaches = len(Xtest)
            Zpred_test = decodingregressor.predict(np.concatenate(Xtest))
        
            idx = 0
            Zpred_test_segmented = []
            for i, z in enumerate(Ztest):
                Zpred_test_segmented.append(Zpred_test[idx:idx+z.shape[0]])
                idx += z.shape[0]

            assert(np.all([z1.shape[0] == z2.shape[0] for (z1, z2) in zip(Zpred_test_segmented, Ztest)]))

        else:
            Xtest = None
            Ztest = None
            num_test_reaches = 0

    if include_velocity and include_acc:

        # Additionally calculate the individual MSE. Do not average over data points
        mse_train = [(z1 - z2)**2 for (z1, z2) in zip(Zpred_segmented, Ztrain)]
        Ztrain = np.concatenate(Ztrain)
        lr_r2_pos = r2_score(Ztrain[..., 0:behavior_dim], Zpred[..., 0:behavior_dim])
        lr_r2_vel = r2_score(Ztrain[..., behavior_dim:2*behavior_dim], Zpred[..., behavior_dim:2*behavior_dim])
        lr_r2_acc = r2_score(Ztrain[..., 2*behavior_dim:], Zpred[..., 2*behavior_dim:])

        if Xtest is not None:
            mse_test = [(z1 - z2)**2 for (z1, z2) in zip(Zpred_test_segmented, Ztest)]
            Ztest = np.concatenate(Ztest)
            lr_r2_post = r2_score(Ztest[..., 0:behavior_dim], Zpred_test[..., 0:behavior_dim])
            lr_r2_velt = r2_score(Ztest[..., behavior_dim:2*behavior_dim], Zpred_test[..., behavior_dim:2*behavior_dim])
            lr_r2_acct = r2_score(Ztest[..., 2*behavior_dim:], Zpred_test[..., 2*behavior_dim:])
        else:
            mse_test = np.nan
            lr_r2_post = np.nan
            lr_r2_velt = np.nan
            lr_r2_acct = np.nan
        return lr_r2_pos, lr_r2_vel, lr_r2_acc, lr_r2_post, lr_r2_velt, lr_r2_acct, decodingregressor, num_test_reaches, fi1, fi2, mse_train, mse_test 

    elif include_velocity:
        raise NotImplementedError
    else:
        raise 

############################### Residual decoding #########################################
def decorrelate(Z, decorrelation='entire', embed=True, transition_times=None, window_indices=None):
    print('Decorrelating!')
    if decorrelation == 'entire':
        pos = Z[:, 0:2]
        vel = Z[:, 2:4]
        acc = Z[:, 4:]

        posdecodingregressor = LinearRegression(fit_intercept=True)
        pos_vel = np.hstack([pos, vel])
        posdecodingregressor.fit(pos, vel)

        # Extract the residuals
        vel_residuals = vel - posdecodingregressor.predict(pos)

        posveldecodingregressor = LinearRegression(fit_intercept=True)
        posveldecodingregressor.fit(pos_vel, acc)
        # Extract the residuals
        acc_residuals = acc - posveldecodingregressor.predict(pos_vel)
    elif decorrelation == 'trialized':

        # Segment Z by transition times
        Z_segmented = [Z[t[0]:t[1]] for t in transition_times]

        # Concatenate, predict, and re-segment
        vel_residuals, acc_residuals = decorrelate(np.concatenate(Z_segmented), decorrelation='entire')

        transition_lengths = np.cumsum([t[1] - t[0] for t in transition_times])
        transition_lengths = np.concatenate([[0], transition_lengths]).astype(int)

        vel_segmented = [vel_residuals[transition_lengths[i]:transition_lengths[i + 1]] 
                        for i in range(len(transition_lengths) - 1)]
        acc_segmented = [acc_residuals[transition_lengths[i]:transition_lengths[i + 1]] 
                        for i in range(len(transition_lengths) - 1)]
        # assert(np.all([v.shape[0] == t[1] - t[0] for (v, t) in zip(vel_segmented, transition_times)]))

        if embed:
            vel_residuals = deepcopy(Z)[:, 2:4] 
            acc_residuals = deepcopy(Z)[:, 4:]

            for i, t in enumerate(transition_times):
                vel_residuals[t[0]:t[1]] = vel_segmented[i]
                acc_residuals[t[0]:t[1]] = acc_segmented[i]
        else:
            vel_residuals = vel_segmented
            acc_residuals = acc_segmented


    elif decorrelation == 'trialized_windowed':

        # In this case, the segmentation and windowing has already been done for us
        Z = np.concatenate(Z)
        vel_residuals, acc_residuals = decorrelate(Z, decorrelation='entire')
        # Need to re-apply the windowing/segmentation
        # For windows later in the transition period, we will have some window indices that only contain
        # a single element

        transition_lengths = np.cumsum([len(w) for w in window_indices])
        transition_lengths = np.concatenate([[0], transition_lengths]).astype(int)
        vel_residuals = [vel_residuals[transition_lengths[i]:transition_lengths[i + 1]] 
                        for i in range(len(transition_lengths) - 1)]
        acc_residuals = [acc_residuals[transition_lengths[i]:transition_lengths[i + 1]] 
                        for i in range(len(transition_lengths) - 1)]

    return vel_residuals, acc_residuals

# Sequentually regress position onto velocity, predict the reisudals, and then regress position and 
# acceleration onto acceleration, and then predict the residuals
def lr_residual_decoder(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, decoding_window=1):

    behavior_dim = Ztrain[0].shape[-1]

    Xtest, Xtrain, Ztest, Ztrain = lr_preprocess(Xtest, Xtrain, Ztest, Ztrain, trainlag, testlag, 
                                                 decoding_window, include_velocity=True, include_acc=True)
    # No train test split here
    vel_residuals, acc_residuals = decorrelate(np.hstack([Ztrain, Ztest]), decorrelation='entire')

    velresidual_decoder = LinearRegression(fit_intercept=True)
    velresidual_decoder.fit(Xtrain, vel_residuals[:Xtrain.shape[0], :])
    vel_residuals_pred = velresidual_decoder.predict(Xtest)
    lr_r2_vel = r2_score(vel_residuals[Xtrain.shape[0]:, :], vel_residuals_pred)

    accresidual_decoder = LinearRegression(fit_intercept=True)
    accresidual_decoder.fit(Xtrain, acc_residuals[:Xtrain.shape[0], :])
    acc_residuals_pred = accresidual_decoder.predict(Xtest)
    lr_r2_acc = r2_score(acc_residuals[Xtrain.shape[0]:, :], acc_residuals_pred)       

    return np.nan, lr_r2_vel, lr_r2_acc, np.nan, np.nan, np.nan, np.nan

def apply_window_residual(X, Z, lag, window, transition_times, decoding_window, include_velocity, include_acc, 
                 subset_selection=None, offsets=None, enforce_full_indices=False, decorrelation='entire'):

    include_acc = True

    # Apply decoding window
    X = form_lag_matrix(X, decoding_window)
    Z = Z[decoding_window//2:, :]
    Z = Z[:X.shape[0], :]

    # Expand state space to include velocity and acceleration
    if np.any([include_velocity, include_acc]):
        Z, X = expand_state_space([Z], [X], include_velocity, include_acc)

        # Flatten list structure imposed by expand_state_space
        Z = Z[0]
        X= X[0]

    # This *also* requires shifting the transition times, as behavior will have been affected
    # There is a shift due to the formation of the lag matrix *and* the expansion of the state sapce, 
    # which will cut off the first 2 sample points, if include_acc is set to true

    if decoding_window > 1:
        transition_times = [(t[0] - decoding_window//2, t[1] - decoding_window//2) for t in transition_times]        
    try:
        assert(X.shape[0] == Z.shape[0])
    except:
        pdb.set_trace()

    if include_acc:
        transition_times = [(t[0] - 2, t[1] - 2) for t in transition_times]
    elif include_velocity:
        transition_times = [(t[0] - 1, t[1] - 1) for t in transition_times]

    # We assume that the subset indices have already been shifted (this is the case in biasvariance_vst)

    # Decorrelate the time series according to the desired decorrelation method
    # Windowed decorrelation is done after windowing
    if decorrelation in ['entire', 'trialized']:
        vel_residuals, acc_residuals = decorrelate(Z, decorrelation=decorrelation, transition_times=transition_times)

        Z = np.hstack([vel_residuals, acc_residuals])

    # Segment the time series with respect to the transition times (including lag)
    xx = []
    zz = []

    valid_idxs = []
    # Which reaches had no truncation due to start of the next reach?
    full_idxs = []

    # If given a single window, duplicate it across all transition times
    if len(window) != len(transition_times):
        window = [window for _ in range(len(transition_times))]

    # If no offsets provided, let it be 0 for all transition times
    if offsets is None:
        offsets = np.zeros(len(transition_times))

    # Needed for decorrelation
    window_indices_all = []

    for i, (t0, t1) in enumerate(transition_times):
        for j, win in enumerate(window[i]):
            window_indices = np.arange(t0 + win[0] + offsets[i], t0 + win[1] + offsets[i])
            if subset_selection is not None:
                # Select only indices that do not belong to the first velocity peak
                subset_indices = np.arange(t0, t1)[subset_selection[i]]
                window_indices = np.intersect1d(window_indices, subset_indices)

            # No matter what, we should remove segments that overlap with the next transition
            if i < len(transition_times) - 1:
                l1 = len(window_indices)
                window_indices = window_indices[window_indices < transition_times[i + 1][0]]
                if len(window_indices) == l1:
                    full_idxs.append(i)
            else:
                # Or else make sure that we don't exceed the length of the time series
                window_indices = window_indices[window_indices < Z.shape[0]]

            window_indices = window_indices.astype(int)
            zz_ = Z[window_indices]
            # Shift x indices by lag
            window_indices -= lag
            xx_ = X[window_indices]

            if len(xx_) > 0:
                assert(xx_.shape[0] == zz_.shape[0])
                if enforce_full_indices:
                    if i in full_idxs:
                        xx.append(xx_)
                        zz.append(zz_)
                else:
                    xx.append(xx_)
                    zz.append(zz_)
                
                window_indices_all.append(window_indices)

                # try:
                #     assert(full_idxs[-1] in valid_idxs or i == len(transition_times) - 1)
                # except:
                #     pdb.set_trace()

    if decorrelation == 'trialized_windowed':
        vel_residuals, acc_residuals = decorrelate(zz, decorrelation='trialized_windowed', 
                                                   window_indices=window_indices_all)
        zz = [np.hstack([v, a]) for (v, a) in zip(vel_residuals, acc_residuals)]
    return xx, zz, valid_idxs, full_idxs

def lr_residual_decode_windowed(X, Z, lag, train_windows, test_windows, transition_times, 
                                train_idxs, test_idxs=None, 
                                decoding_window=1, include_velocity=True, include_acc=True, 
                                pkassign=None, apply_pk_to_train=False, offsets=None,
                                decorrelation='entire'):

    # There are 3 versions of this - decorrelation of the entire time series...
    # Decorrelation of the trialized time series
    # Decorrelation of each windowed segment

    behavior_dim = Z.shape[-1]

    # We have been given a list (of list) of windows for each transition
    win_min = train_windows[0][0]

    if win_min >= 0:
        win_min = 0

    # Filter out by transitions that lie within the train idxs, and stay clear of the start and end
    tt_train = [(t, idx) for idx, t in enumerate(transition_times) 
                if idx in train_idxs and t[0] > (lag + np.abs(win_min)) and t[1] < (Z.shape[0] - lag - np.abs(win_min))]
    # Re-assign train idxs removing those reaches that were outside the start/end region
    train_idxs = [x[1] for x in tt_train]
    tt_train = [x[0] for x in tt_train]

    if offsets is not None:
        offsets_train = offsets[train_idxs]
    else:
        offsets_train = None

    if apply_pk_to_train:
        # Train on the first velocity peak only
        assert(np.all([s.size == np.arange(t[0], t[1]).size for (s, t) in zip(pkassign[train_idxs], tt_train)]))
        subset_selection = [np.argwhere(np.array(s) == 0).squeeze() for s in pkassign[train_idxs]]
        Xtrain, Ztrain, vi1, fi1 = apply_window_residual(X, Z, lag, train_windows, tt_train, decoding_window, include_velocity, include_acc, subset_selection, offsets=offsets_train,
                                                         enforce_full_indices=True, decorrelation=decorrelation)

    else:
        Xtrain, Ztrain, vi1, fi1 = apply_window_residual(X, Z, lag, train_windows, tt_train, decoding_window, include_velocity, include_acc, offsets=offsets_train,
                                                         enforce_full_indices=True, decorrelation=decorrelation)

    if Xtrain is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None, 0
    else:
        n = len(Xtrain)

    if test_idxs is not None:
        # Filter out by transitions that lie within the test idxs, and stay clear of the start and end
        tt_test = [(t, idx) for idx, t in enumerate(transition_times) 
                   if idx in test_idxs and t[0] > (lag + np.abs(win_min)) and t[1] < (Z.shape[0] - lag - np.abs(win_min))]
        # Re-assign test idxs removing those reaches that were outside the start/end region
        test_idxs = [x[1] for x in tt_test]
        tt_test = [x[0] for x in tt_test]


        if offsets is not None:
            offsets_test = offsets[test_idxs]
        else:
            offsets_test = None

        if pkassign is not None:
            assert(np.all([s.size == np.arange(t[0], t[1]).size for (s, t) in zip(pkassign[test_idxs], tt_test)]))
            subset_selection = [np.argwhere(np.array(s) != 0).squeeze() for s in pkassign[test_idxs]]
            Xtest, Ztest, _, fi2 = apply_window_residual(X, Z, lag, test_windows, tt_test, decoding_window, include_velocity, include_acc, subset_selection, offsets=offsets_test, 
                                               enforce_full_indices=True, decorrelation=decorrelation)        
        else:
            Xtest, Ztest, _, fi2 = apply_window_residual(X, Z, lag, test_windows, tt_test, decoding_window, include_velocity, include_acc, offsets=offsets_test,
                                               enforce_full_indices=True, decorrelation=decorrelation)

    else:
        Xtest = None
        Ztest = None

    # Standardize
    # X = StandardScaler().fit_transform(X)
    # Z = StandardScaler().fit_transform(Z)
    decodingregressor = LinearRegression(fit_intercept=True)

    # Fit and score
    if len(Xtrain) == 0:
        return tuple([np.nan] * 9) + (0,)
    decodingregressor.fit(np.concatenate(Xtrain), np.concatenate(Ztrain))
    Zpred = decodingregressor.predict(np.concatenate(Xtrain))

    # Re-segment Zpred
    idx = 0
    Zpred_segmented = []
    for i, z in enumerate(Ztrain):
        Zpred_segmented.append(Zpred[idx:idx+z.shape[0]])
        idx += z.shape[0]

    assert(np.all([z1.shape[0] == z2.shape[0] for (z1, z2) in zip(Zpred_segmented, Ztrain)]))
    #Ztrain = np.concatenate(Ztrain)
    if Xtest is not None:
        if len(Xtest) > 0:
            num_test_reaches = len(Xtest)
            Zpred_test = decodingregressor.predict(np.concatenate(Xtest))
        
            idx = 0
            Zpred_test_segmented = []
            for i, z in enumerate(Ztest):
                Zpred_test_segmented.append(Zpred_test[idx:idx+z.shape[0]])
                idx += z.shape[0]

            assert(np.all([z1.shape[0] == z2.shape[0] for (z1, z2) in zip(Zpred_test_segmented, Ztest)]))

        else:
            Xtest = None
            Ztest = None
            num_test_reaches = 0


    # Additionally calculate the individual MSE. Do not average over data points
    mse_train = [(z1 - z2)**2 for (z1, z2) in zip(Zpred_segmented, Ztrain)]
    Ztrain = np.concatenate(Ztrain)
    lr_r2_vel = r2_score(Ztrain[..., 0:2], Zpred[..., 0:2])
    lr_r2_acc = r2_score(Ztrain[..., 2:], Zpred[..., 2:])

    if Xtest is not None:
        mse_test = [(z1 - z2)**2 for (z1, z2) in zip(Zpred_test_segmented, Ztest)]
        Ztest = np.concatenate(Ztest)
        lr_r2_velt = r2_score(Ztest[..., 0:2], Zpred_test[..., 0:2])
        lr_r2_acct = r2_score(Ztest[..., 2:], Zpred_test[..., 2:])
    else:
        mse_test = np.nan
        lr_r2_post = np.nan
        lr_r2_velt = np.nan
        lr_r2_acct = np.nan
    return np.nan, lr_r2_vel, lr_r2_acc, np.nan, lr_r2_velt, lr_r2_acct, decodingregressor, num_test_reaches, fi1, fi2, mse_train, mse_test 

def logreg_preprocess(x_in, y_in):

    if isinstance(x_in, np.ndarray):
        n_trials = x_in.shape[0]
        if x_in.ndim == 3:
            n_dof = x_in.shape[-1]
            x_in = [x for x in x_in]
        else:
            n_dof = x_in[0].shape[-1]
    else:
        n_trials = len(x_in)
        n_dof = x_in[0].shape[-1]

    x_out = np.zeros((n_trials, n_dof))
    for trial in np.arange(n_trials):
        x_out[trial, :] = np.sum(np.squeeze(x_in[trial]), 0)
    scaler = StandardScaler()
    x_out = scaler.fit_transform(x_out)

    return x_out, y_in

def rrglm_decoder(Xtest, Xtrain, Ztest, Ztrain, rank=None):

    Xtrain, Ztrain = logreg_preprocess(Xtrain, Ztrain)
    Xtest, Ztest = logreg_preprocess(Xtest, Ztest)
    # Convert training data to an R dataframe
    data = {'x%d' % i: Xtrain[:, i] for i in range(Xtrain.shape[1])}
    # map z to integers 1, M + 1
    data['z'] = LabelEncoder().fit_transform(Ztrain) + 1
    df = pd.DataFrame(data)
    pandas2ri.activate()
    r_df = pandas2ri.py2rpy(df)
    # Load the VGAM package
    vgam = importr('VGAM')
    base = importr('base')
    # glm = importr('glm')
    # All variables except for z
    formula = ro.Formula('z ~ .')
    if rank is None:
        vglm_model = vgam.vglm(formula, family=vgam.multinomial(), data=r_df, trace=True)
    else:
        vglm_model = vgam.rrvglm(formula, family=vgam.multinomial(), data=r_df, Rank=rank)

    predictions_r = ro.r['predict'](vglm_model, newdata=test_df, type="response")

    # Convert R object to numpy array or pandas DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        predictions_np = ro.conversion.rpy2py(predictions_r)

def logreg(Xtest_in, Xtrain_in, Ytest, Ytrain, opt_args=False):

    Xtest_in, Xtrain_in, Ytest, Ytrain = np.array(Xtest_in), np.array(Xtrain_in), np.array(Ytest), np.array(Ytrain)

    # For the Tsao data, we're not going to time resolve:
    Xtrain = np.zeros((np.shape(Xtrain_in)[0], np.shape(Xtrain_in)[2]))
    Xtest = np.zeros((np.shape(Xtest_in)[0], np.shape(Xtest_in)[2]))

    for trial in np.arange(np.shape(Xtrain_in)[0]):
        Xtrain[trial, :] = np.sum(np.squeeze(Xtrain_in[trial, :, :]), 0)
    
    for trial in np.arange(np.shape(Xtest_in)[0]):
        Xtest[trial, :] = np.sum(np.squeeze(Xtest_in[trial, :, :]), 0)

    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)


    # Run actual decoder
    clf = LogisticRegression(multi_class="multinomial").fit(Xtrain, Ytrain)
    predictions = clf.predict(Xtest)
    loss = hamming_loss(Ytest, predictions)

    
    # Ensure compatibility with batch_analysis
    return {"predictions":predictions, "loss":loss}, None, None, None, None, None, None
