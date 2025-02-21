import glob
import os
import numpy as np
from sklearn.model_selection import KFold

#script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
script_path = '/home/akumar/nse/neural_control/batch_analysis.py'

desc = 'Fits using variable window width filtering'
#desc = 'Fits of dimreduc methods to loco data'

#data_path = os.environ['SCRATCH'] + '/sabes'
data_path = '/mnt/Secondary/data/sabes'    
 
# These are the data files that contain both M1 and S1 recordings.
data_files = glob.glob('%s/indy*' % data_path)
# Fit on only 5 data files
data_files = [data_files[idx] for idx in [0, 5, 10, 15, 20, 25]]

#data_files = glob.glob('%s/loco*' % data_path)
#data_files = [data_files[0], data_files[5]]
  
 # Load the data files and determine how many dof (neurons) there are in each recording
 # data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]
 
loader = 'sabes'
analysis_type = 'dimreduc'
 
 # Each of these can be made into a list whose outer product is taken
loader_args = [{'bin_width':100, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1'},
               {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1'},
               {'bin_width':25, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1'}]

for bin_width in [25, 50, 100]:
    for window_length in [5, 10, 20]:
        loader_args.append({'bin_width':bin_width, 'filter_fn':'window', 'filter_kwargs':{'window_name':'hann', 'window_length':window_length}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1'})

n_folds=5
# loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100}]
dimvals = np.array([2, 4, 6, 10])
task_args = [{'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':10}},
             {'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'PCA', 'dimreduc_args': {}}]
