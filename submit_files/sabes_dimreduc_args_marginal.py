import glob
import os
import pickle
import numpy as np
from sklearn.model_selection import KFold

#script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
script_path = '/home/akumar/nse/neural_control/batch_analysis.py'

desc = 'Dimreduc with start time modification and subset selection'
#desc = 'Fits of dimreduc methods to loco data'

#data_path = os.environ['SCRATCH'] + '/sabes'
data_path = '/mnt/Secondary/data/sabes'    
 
#data_files = glob.glob('%s/loco*.mat' % data_path)
data_files = glob.glob('%s/indy*.mat' % data_path)
good_loco_files = ['loco_20170210_03.mat',
            'loco_20170213_02.mat',
            'loco_20170215_02.mat',
            'loco_20170227_04.mat',
            'loco_20170228_02.mat',
            'loco_20170301_05.mat',
            'loco_20170302_02.mat']

for glf in good_loco_files:
    data_files.append('%s/%s' % (data_path, glf))

# Only fit every third data file to save time
loader = 'sabes'
analysis_type = 'dimreduc'

# Only perform dimreduc on neurons with low FBC importance score
with open('/home/akumar/nse/neural_control/subset_indices.pkl', 'rb') as f:
    subset_indices = pickle.load(f)

loader_args = [{'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1',
                'truncate_start':True, 'subset':subset_indices},
                {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1',
                'truncate_start':False, 'subset':subset_indices},
                {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1',
                'truncate_start':True, 'subset':None},
                {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100, 'region':'M1',
                'truncate_start':False, 'subset':None}]
dimvals = np.arange(1, 31)
task_args = [{'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'PCA', 'dimreduc_args': {'marginal_only':True}},
             {'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'LQGCA', 
              'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42, 'marginal_only':True}}]