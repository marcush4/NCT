import pdb
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import pickle
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from dca.methods_comparison import JPCA 
from config import PATH_DICT
from region_select_alt import *
 
from fig6calcs import get_rates, dim_dict, T_dict
 
def get_random_projections(region):
    random_proj_path = PATH_DICT['tmp'] + '/jpca_tmp_randcontrol_%s.pkl' % region
    with open(random_proj_path, 'rb') as f:
        control_results = pickle.load(f)
    controldf = pd.DataFrame(control_results)

    return controldf

def get_df(region):
    path = PATH_DICT['tmp'] + '/jpca_tmp_%s.pkl' % region
    with open(path, 'rb') as f:
        results = pickle.load(f)
    df = pd.DataFrame(results)
    return df

def make_rot_plots(x, df_fcca, df_pca, maxim, region, jDIM, figpath):


    xpca = x @ df_pca.iloc[0]['coef'][:, 0:jDIM]
    xdca = x @ df_fcca.iloc[0]['coef']

    jpca1 = JPCA(n_components=jDIM, mean_subtract=False)
    jpca1.fit(xpca)

    jpca2 = JPCA(n_components=jDIM, mean_subtract=False)
    jpca2.fit(xdca)

    xpca_j = jpca1.transform(xpca)
    xdca_j = jpca2.transform(xdca)

    xpca_j_mean = np.mean(xpca_j, 0).squeeze()
    xdca_j_mean = np.mean(xdca_j, 0).squeeze()

    ################################### PLOT CODE ##############################

    # Save as two separate figures
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
    ax = [ax1, ax2]

    for i in range(0, 25):
        
        """" 
        For trialized data: Will require something like: 
        datpath = '/mnt/Secondary/data/sabes'
        dat = load_sabes('%s/%s' % (datpath, data_file))
        dat = reach_segment_sabes(dat, data_file=data_file.split('.mat')[0])


            ......

            pca_straightdev = np.zeros(len(dat['target_pairs']))
            dca_straightdev = np.zeros(len(dat['target_pairs']))
            transition_times = dat['transition_times']
            for i in range(len(dat['target_pairs'])):
                
                trajectory = gaussian_filter1d(xpca_j[0, transition_times[i][0]:transition_times[i][1]], 
                                            sigma=5, axis=0)
                trajectory -= trajectory[0]
                        
                start = trajectory[0, :]
                end = trajectory[-1, :]
                
                pca_straightdev[i] = measure_straight_dev(trajectory, start, end)

                trajectory = gaussian_filter1d(xdca_j[0, transition_times[i][0]:transition_times[i][1]], 
                                            sigma=5, axis=0)
                trajectory -= trajectory[0]
                
                start = trajectory[0, :]
                end = trajectory[-1, :]
                dca_straightdev[i] = measure_straight_dev(trajectory, start, end)

            pca_devorder = np.argsort(pca_straightdev)[::-1]
            dca_devorder = np.argsort(dca_straightdev)[::-1]

            idx = dca_devorder[i]
            t0 = transition_times[idx][0]
            t1 = min(transition_times[idx][0] + 40, transition_times[idx][1])
            trajectory = gaussian_filter1d(xpca_j[0, t0:t1],  sigma=5, axis=0)[:-3]
        """
        trajectory = gaussian_filter1d(xpca_j[i,:,:].squeeze(),  sigma=4, axis=0)

        # Center and normalize trajectories
        trajectory -= trajectory[0]
        #trajectory /= np.linalg.norm(trajectory)

        # Rotate trajectory so that the first 5 timesteps all go off at the same angle
        theta0 = np.arctan2(trajectory[15, 1], trajectory[15, 0])

        # Rotate *clockwise* by theta
        R = lambda theta: np.array([[np.cos(-1*theta), -np.sin(-theta)], \
                                    [np.sin(-theta), np.cos(theta)]])        
        trajectory = np.array([R(theta0 - np.pi/4) @ t[0:2] for t in trajectory])

        ax[1].plot(trajectory[:, 0], trajectory[:, 1], 'k', alpha=0.5)
        ax[1].arrow(trajectory[-1, 0], trajectory[-1, 1], 
                    trajectory[-1, 0] - trajectory[-2, 0], trajectory[-1, 1] - trajectory[-2, 1], 
                    head_width=0.08, color="k", alpha=0.5)
        
        """" 
        idx = dca_devorder[i]
        t0 = transition_times[idx][0]
        t1 = min(transition_times[idx][0] + 40, transition_times[idx][1])
        trajectory = gaussian_filter1d(xdca_j[0, t0:t1], sigma=5, axis=0)[:-3]
        """
        trajectory = gaussian_filter1d(xdca_j[i,:,:].squeeze(),  sigma=4, axis=0)

        # Center trajectories
        trajectory -= trajectory[0]
        #trajectory /= np.linalg.norm(trajectory)

        # Rotate trajectory so that the first 5 timesteps all go off at the same angle
        theta0 = np.arctan2(trajectory[15, 1], trajectory[15, 0])

        trajectory = np.array([R(theta0 - np.pi/4) @ t[0:2] for t in trajectory])

        ax[0].plot(trajectory[:, 0], trajectory[:, 1], '#c73d34', alpha=0.5)
        ax[0].arrow(trajectory[-1, 0], trajectory[-1, 1], 
                    trajectory[-1, 0] - trajectory[-2, 0], trajectory[-1, 1] - trajectory[-2, 1], 
                    head_width=0.05, color="#c73d34", alpha=0.5)

    _, p = scipy.stats.wilcoxon(maxim[:, 0, 2], maxim[:, 1, 2], alternative='less')
   # print('Re p:%f' % p)


    ax[0].set_aspect('equal')   
    ax[1].set_aspect('equal')
    #ax[1].set_xlim([-3.2, 5.3])
    #ax[1].set_ylim([-3.2, 5.3])

    #ax[0].set_xlim([-2.2, 4])
    #ax[0].set_ylim([-2.2, 4])

    ax[0].spines['right'].set_color('none')
    ax[0].spines['top'].set_color('none')
    ax[0].spines['left'].set_position('zero')
    ax[0].spines['bottom'].set_position('zero')
    ax[0].plot(2, 0, ">k", clip_on=False)
    ax[0].plot(0, 2, "^k", clip_on=False)
    ax[0].spines['left'].set_bounds(0, 2)
    ax[0].spines['bottom'].set_bounds(0, 2)

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[1].spines['right'].set_color('none')
    ax[1].spines['top'].set_color('none')
    ax[1].spines['left'].set_position('zero')
    ax[1].spines['bottom'].set_position('zero')
    ax[1].spines['left'].set_bounds(0, 2)
    ax[1].spines['bottom'].set_bounds(0, 2)
    ax[1].plot(2, 0, ">k", clip_on=False)
    ax[1].plot(0, 2, "^k", clip_on=False)

    fig1.tight_layout()
    save_fig_path = '%s/trajectories_a_%s.pdf' % (figpath, region)
    fig1.savefig(save_fig_path, bbox_inches='tight', pad_inches=0)


    fig2.tight_layout()
    save_fig_path = '%s/trajectories_b_%s.pdf' % (figpath, region)
    fig2.savefig(save_fig_path, bbox_inches='tight', pad_inches=0)



def make_traj_amplification_plots(df_, y, DIM, region, figpath):

    dimreduc_method = df_['dimreduc_method'].iloc[0]
    if region in ['HPC']:
        methods = np.array(['PCA', 'FCCA'])
    else:
        methods = np.array(['PCA', 'LQGCA'])

    cInd = np.argwhere(methods == dimreduc_method)[0][0]
    colors = ['k', 'r']

    assert(df_.shape[0] == 1)
    V = df_.iloc[0]['coef']
    V = V[:, 0:DIM]        
    # Project data
    yproj = y @ V
    #yproj = np.array([yproj[t0:t0+40] for t0, t1 in dat['transition_times'] if t1 - t0 > 40])
    yproj = np.array([y_ - y_[0] for y_ in yproj])
    dY = np.concatenate(np.diff(yproj, axis=1), axis=0)
    Y_prestate = np.concatenate(yproj[:, :-1], axis=0)

    # Least squares
    A, _, _, _ = np.linalg.lstsq(Y_prestate, dY, rcond=None)
    _, s, _ = np.linalg.svd(A)
    # Iterate the lyapunov equation for 10 timesteps
    P = np.zeros((DIM, DIM))
    for _ in range(10):
        dP = A @ P + P @ A.T + np.eye(DIM)
        P += dP

    eig, U = np.linalg.eig(P)
    # eig, U = np.linalg.eig(scipy.linalg.expm(A.T) @ scipy.linalg.expm(A))
    eig = np.sort(eig)[::-1]
    U = U[:, np.argsort(eig)[::-1]]
    U = U[:, 0:2]
    # Plot smoothed, centered trajectories for all reaches in top 2 dimensions

    # Argsort by the maximum amplitude in the top 2 dimensions
    #trajectory = gaussian_filter1d(yproj, sigma=5, axis=1)
    trajectory = gaussian_filter1d(yproj, sigma=2, axis=1)

    trajectory -= trajectory[:, 0:1, :]
    trajectory = trajectory @ U
    dyn_range = np.max(np.abs(trajectory), axis=1)
    #dyn_range = np.max(np.abs(yproj @ U), axis=1)
    ordering = np.argsort(dyn_range, axis=0)[::-1]

    #t0 = trajectory[ordering[:, 0], :, 0]
    #t1 = trajectory[ordering[:, 1], :, 1]
    t0 = trajectory[:, :, 0]
    t1 = trajectory[:, :, 1]

    f1, a1 = plt.subplots(1, 1, figsize=(4.2, 4))
    f2, a2 = plt.subplots(1, 1, figsize=(4.2, 4))
    ax = [a1, a2]

    for i in range(min(50, t0.shape[0])):
        ax[0].plot(np.arange(len(t0[i])), t0[i], color=colors[cInd], alpha=0.5, linewidth=1.5)
        ax[1].plot(np.arange(len(t1[i])), t1[i], color=colors[cInd], alpha=0.5, linewidth=1.5)
        #ax[2*j].set_title(np.sum(eig))
        
    for a in ax:
        a.spines['bottom'].set_position('center')
        # Eliminate upper and right axes
        a.spines['right'].set_color('none')
        a.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        a.xaxis.set_ticks_position('bottom')
        a.yaxis.set_ticks_position('left')

        a.set_xticks([0, 2])
        a.set_xticklabels([])
        a.tick_params(axis='both', labelsize=12)

        a.set_xlabel('Time (s)', fontsize=12)
        a.xaxis.set_label_coords(1.1, 0.56)
        
    # Set y scale according to the current yscale on PCA 0
    if cInd == 0:
        ylim_max = np.max(np.abs(t0[0])) + 1.5
        #ylim = [-ylim_max, ylim_max]

    for a in ax:
        #a.set_ylim(ylim)
        #a.set_yticks([-int(ylim_max), 0, int(ylim_max)])
        a.set_ylabel('Amplitude (a.u.)', fontsize=12)

    if cInd == 0:
        ax[0].set_title('FFC Component 1', fontsize=12)
        ax[1].set_title('FFC Component 2', fontsize=12)
    else:
        ax[0].set_title('FBC Component 1', fontsize=12)
        ax[1].set_title('FBC Component 2', fontsize=12)

    
    f1.tight_layout()
    save_fig_path = '%s/amplifications_comp1_%s_%s.pdf' % (figpath, dimreduc_method, region)
    f1.savefig(save_fig_path, bbox_inches='tight', pad_inches=0)

    f2.tight_layout()
    save_fig_path = '%s/amplifications_comp2_%s_%s.pdf' % (figpath, dimreduc_method, region)
    f2.savefig(save_fig_path, bbox_inches='tight', pad_inches=0)



def make_box_plots(maxim, maxim_control, region, figpath):
        
    # Boxplots
    fig0, ax0 = plt.subplots(figsize=(6, 3))

    medianprops = dict(linewidth=1, color='b')
    whiskerprops=dict(linewidth=0)

    # Center relative to random - per recording session
    mu = np.mean(maxim_control[..., 1], axis=1)
    sigma = np.std(maxim_control[..., 1], axis=1)
    r1 = maxim[:, 0, 1] - mu
    r2 = maxim[:, 1, 1] - mu

    bplot = ax0.boxplot([r1, r2], patch_artist=True, 
                    medianprops=medianprops, notch=False, vert=False, showfliers=False, widths=[0.3, 0.3],
                    whiskerprops=whiskerprops, showcaps=False)
    
    

    _, p = scipy.stats.wilcoxon(maxim[:, 0, 1], maxim[:, 1, 1], alternative='greater')


    # test that each is stochastically greater than the median random
    x1 = maxim[:, 0, 1] - np.median(maxim_control[..., 1].ravel())
    x2 = maxim[:, 1, 1] - np.median(maxim_control[..., 1].ravel())

    _, p1 = scipy.stats.wilcoxon(x1, alternative='greater')
    _, p2 = scipy.stats.wilcoxon(x2, alternative='greater')


    method1 = 'FBC'
    method2 = 'FFC'

    ax0.set_yticklabels([method1, method2], fontsize=12)
    #ax0.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
    #ax0.set_xlim([0, 0.2])
    ax0.tick_params(axis='both', labelsize=12)
    #ax.set_ylabel(r'$\sum_i Im(\lambda_i)$', fontsize=22)
    ax0.set_xlabel('Rotational Strength above Random', fontsize=12)
    #ax.set_title('****', fontsize=14)

    ax0.invert_yaxis()

    fig0.tight_layout()
    save_fig_path = '%s/rot_strength_box_plots_%s.pdf' % (figpath, region)
    fig0.savefig(save_fig_path, bbox_inches='tight', pad_inches=0)


    fig1, ax1 = plt.subplots(figsize=(6, 3))

    # fill with colors
    colors = ['red', 'black', 'blue']   
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    whiskerprops = dict(linewidth=0)

    mu = np.mean(maxim_control[..., 2], axis=1)
    sigma = np.std(maxim_control[..., 2], axis=1)
    r1 = maxim[:, 0, 2] - mu
    r2 = maxim[:, 1, 2] - mu

    bplot = ax1.boxplot([r1, r2], patch_artist=True, 
                    medianprops=medianprops, notch=False, vert=False, showfliers=False, widths=[0.3, 0.3],
                    whiskerprops=whiskerprops, showcaps=False)

    x1 = maxim[:, 0, 2] - np.median(maxim_control[..., 2].ravel())
    x2 = maxim[:, 1, 2] - np.median(maxim_control[..., 2].ravel())

    _, p1 = scipy.stats.wilcoxon(x1, alternative='greater')
    _, p2 = scipy.stats.wilcoxon(x2, alternative='greater')


    method1 = 'FBC'
    method2 = 'FFC'

    ax1.set_yticklabels([method1, method2], fontsize=12)
    #ax1.set_xticks([0, 1, 2, 3, 4])
    #ax1.set_xlim([0, 4.5])
    ax1.tick_params(axis='both', labelsize=12)
    #ax.set_ylabel(r'$\sum_i Im(\lambda_i)$', fontsize=22)
    ax1.set_xlabel('Dynamic Range above Random', fontsize=12)
    #ax.set_title('****', fontsize=14)

    ax1.invert_yaxis()

    # fill with colors
    colors = ['red', 'black', 'blue']   
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)


    # ax.set_xlim([13, 0])
    fig1.tight_layout()
    save_fig_path = '%s/dyn_rng_box_plots_%s.pdf' % (figpath, region)
    fig1.savefig(save_fig_path, bbox_inches='tight', pad_inches=0)

def plotFig6(decoding_df, session_key, data_path, region, DIM, figpath='.'):
    jDIM = DIM - 1 if DIM % 2 != 0 else DIM    # jPCA dimension must be even

    ################## Load random projections for later comparison
    controldf = get_random_projections(region)
    A_df = get_df(region)

    sessions = np.unique(decoding_df[session_key].values)

    d_U = np.zeros((len(sessions), 2, 3))
    maxim = np.zeros((len(sessions), 2, 3))

    control_reps = len(np.unique(controldf['inner_rep'].values))
    maxim_control = np.zeros((len(sessions), control_reps, 3))

    for i in range(len(sessions)):
        for j, dimreduc_method in enumerate(['LQGCA', 'PCA']):
            df_ = apply_df_filters(A_df, **{session_key: sessions[i], 'dimreduc_method':dimreduc_method})
            eigs = df_.iloc[0]['jeig']
            maxim[i, j, 0] = np.sum(np.abs(eigs))/2
            maxim[i, j, 1] = np.sum(np.abs(eigs))/2
            maxim[i, j, 2] = df_.iloc[0]['dyn_range']

        for j in range(control_reps):
            df_ = apply_df_filters(controldf, **{session_key: sessions[i], 'inner_rep':j})
            assert(df_.shape[0] == 1)

            eigs = df_.iloc[0]['jeig']
            maxim_control[i, j, 0] = np.sum(np.abs(eigs))/2
            maxim_control[i, j, 1] = np.sum(np.abs(eigs))/2
            eigs = df_.iloc[0]['dyn_range']
            maxim_control[i, j, 2] = eigs

    make_box_plots(maxim, maxim_control, region, figpath)
    # Plot projections for the sessions
    for i, session in enumerate(sessions):

        if region in ['AM', 'ML']:
            fcca_filter = {session_key:session, 'fold_idx':0, 'dim':DIM, 'dimreduc_method':'LQGCA', 'loader_args':{'region': region}}
            pca_filter = {session_key:session, 'fold_idx':0, 'dim':DIM, 'dimreduc_method':'PCA', 'loader_args':{'region': region}}
        else:
            fcca_filter = {session_key:session, 'fold_idx':0, 'dim':DIM, 'dimreduc_method':['LQGCA', 'FCCA']}
            pca_filter = {session_key:session, 'fold_idx':0, 'dim':DIM, 'dimreduc_method':'PCA'}

        df_fcca = apply_df_filters(decoding_df, **fcca_filter)
        df_pca = apply_df_filters(decoding_df, **pca_filter)

        y = get_rates(T_dict[region], decoding_df, data_path, region, session)
        make_rot_plots(y, df_fcca, df_pca, maxim, region, jDIM, figpath)
        make_traj_amplification_plots(df_fcca, y, jDIM, region, figpath)
        make_traj_amplification_plots(df_pca, y, jDIM, region, figpath)

if __name__ == '__main__':


    #regions = ['M1', 'S1', 'M1_maze', 'HPC']
    regions = ['AM', 'ML']

    for region in tqdm(regions):
        figpath = PATH_DICT['figs']
        DIM = dim_dict[region]
        df, session_key = load_decoding_df(region, **loader_kwargs[region])
        data_path = get_data_path(region)
        plotFig6(df, session_key, data_path, region, DIM, figpath=figpath)
