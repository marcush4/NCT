import glob
import os
import itertools
import numpy as np
import pdb


desc = 'Dimreduc on new frank lab dataset'

#script_path = '/home/ankit_kumar/neural_control/batch_analysis.py'
script_path = '/home/marcush/projects/neural_control/batch_analysis.py' 

data_path = '/clusterfs/NSDS_data/franklabdata/dataset1'

data_files = [data_path]
loader = 'franklab_new'
analysis_type = 'dimreduc'

# Each of these can be made into a list whose outer product is taken
bin_width = [10, 25, 50, 100]
trialize = [True, False]
spike_threshold = [0, 100]
regions = ['mPFC', 'HPC']
loader_combs = itertools.product(bin_width, trialize, spike_threshold, regions)
loader_args = []
for lArgs in loader_combs:
    bWidth, bTrialize, spike_thresh, region = lArgs
    if bTrialize:
        loader_args.append({'bin_width':bWidth, 'region':region, 'spike_threshold':spike_thresh, 
                            'speed_threshold':False, 'trialize':bTrialize})
    else:
        loader_args.append({'bin_width':bWidth, 'region':region, 'spike_threshold':spike_thresh, 
                            'speed_threshold':False, 'trialize':bTrialize})
        loader_args.append({'bin_width':bWidth, 'region':region, 'spike_threshold':spike_thresh, 
                            'speed_threshold':True, 'trialize':bTrialize})

# loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}]


""""

MARGINALS IS: ON !!!!!

"""

dimvals = np.arange(1, 41)
task_args = [{'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'PCA', 'dimreduc_args': {'marginal_only':True}}]
for T in np.array([1, 3, 5]):
    task_args.append({'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':T, 'loss_type':'trace', 'n_init':10, 'marginal_only':True}})
    