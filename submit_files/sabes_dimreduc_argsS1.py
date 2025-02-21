import glob
import os
import pickle
import numpy as np
from sklearn.model_selection import KFold

#script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
#script_path = '/home/akumar/nse/neural_control/batch_analysis.py'
script_path = '/home/ankit_kumar/neural_control/batch_analysis.py'

desc = 'Dimreduc with start time modification and subset selection with varying quantiles in S1'
#desc = 'Fits of dimreduc methods to loco data'

#data_path = os.environ['SCRATCH'] + '/sabes'
#data_path = '/mnt/Secondary/data/sabes'    
data_path = '/home/ankit_kumar/Data/sabes'
 
#data_files = glob.glob('%s/loco*.mat' % data_path)
fls = ['loco_20170210_03.mat',
 'loco_20170213_02.mat',
 'loco_20170215_02.mat',
 'loco_20170227_04.mat',
 'loco_20170228_02.mat',
 'loco_20170301_05.mat',
 'loco_20170302_02.mat',
 'indy_20160426_01.mat']

data_files = []
for f in fls:
    data_files.append('%s/%s' % (data_path, f))

# Only fit every third data file to save time
loader = 'sabes'
analysis_type = 'dimreduc'

# Only perform dimreduc on neurons with low FBC importance score
subset_files = glob.glob('/home/ankit_kumar/neural_control/subset_indices/*_subset_q*_S1.pkl')

loader_args = []
for fl in subset_files:
    with open(fl, 'rb') as f:
        subset_indices = pickle.load(f)
    loader_args.append({'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'spike_threshold':100,
                'truncate_start':True, 'subset':subset_indices, 'subset_file': fl.split('/')[-1], 'region':'S1'})

dimvals = np.arange(1, 31)
task_args = [{'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'PCA', 'dimreduc_args': {}},
             {'dim_vals':dimvals, 'n_folds':5, 'dimreduc_method':'LQGCA', 
              'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':10, 'rng_or_seed':42}}]
