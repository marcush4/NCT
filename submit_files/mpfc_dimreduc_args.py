import glob
import os
import itertools
import numpy as np
import pdb

#script_path = '/global/homes/a/akumar25/repos/localization/batch_analysis_sabes.py'
#script_path = '/home/akumar/nse/neural_control/batch_analysis.py'
script_path = '/home/ankit_kumar/neural_control/batch_analysis.py'

desc = 'Dimreduc on new frank lab dataset'
#data_path = '/mnt/Secondary/data/peanut'
# data_path = '/home/ankit_kumar/Data/peanut'
# data_files = ['%s/data_dict_peanut_day14.obj' % data_path]
data_path = '/clusterfs/NSDS_data/franklabdata/dataset1'

data_files = [data_path]
loader = 'franklab_new'
analysis_type = 'dimreduc'

# Each of these can be made into a list whose outer product is taken
bw = [10, 25, 50, 100]
trialize = [True, False]
spike_threshold = [0, 100]
loader_combs = itertools.product(bw, trialize, spike_threshold)
loader_args = []
for la in loader_combs:
    b, t, st = la
    if t:
        loader_args.append({'bin_width':b, 'region':'mPFC', 'spike_threshold':st, 
                            'speed_threshold':False, 'trialize':t})
    else:
        loader_args.append({'bin_width':b, 'region':'mPFC', 'spike_threshold':st, 
                            'speed_threshold':False, 'trialize':t})
        loader_args.append({'bin_width':b, 'region':'mPFC', 'spike_threshold':st, 
                            'speed_threshold':True, 'trialize':t})

# loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}]
dimvals = np.arange(1, 31, 2)
task_args = [{'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'PCA', 'dimreduc_args': {}}]
for T in np.array([1, 3, 5]):
    task_args.append({'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':T, 'loss_type':'trace', 'n_init':10}})