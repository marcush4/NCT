import glob
import numpy as np
import pickle
 
# type
analysis_type = 'dimreduc'

# Paths
file_name = 'degraded_only_v4_AnalysisDegraded_230322_214006_Jamie' # optional, for selecting individual sessions
data_path = '/home/marcush/Data/TsaoLabData/split/degraded'  
script_path = '/home/marcush/projects/neural_control/batch_analysis.py'
desc = 'degraded trials only'


#data_files = glob.glob('%s/degraded_only*.pickle' % data_path)
data_files = glob.glob(f'{data_path}/{file_name}.pickle')


file_name_manual_units = 'degraded_only_v4_AnalysisDegraded_230322_214006_Jamie'
manual_keep_inds_path = "/home/marcush/Data/TsaoLabData/split/degraded/keepInds"

# In order to manually select units ::
with open(f'{manual_keep_inds_path}/{file_name_manual_units}_keep_inds_ML.pkl', 'rb') as f:
    ML_unit_inds = pickle.load(f)
with open(f'{manual_keep_inds_path}/{file_name_manual_units}_keep_inds_AM.pkl', 'rb') as f:
    AM_unit_inds = pickle.load(f)
    
# In order to manually select trials ::
#with open(f'{manual_keep_inds_path}/{file_name_manual_units}_deg_trial_keep_inds.pkl', 'rb') as f:
#    trial_inds = pickle.load(f)
with open(f'{manual_keep_inds_path}/{file_name_manual_units}_clear_trial_keep_inds.pkl', 'rb') as f:
    trial_inds = pickle.load(f)   

# loader args
loader = 'tsao'
loader_args = [{'bin_width':50, 'same_trial_dur':True, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'ML', 'manual_unit_selection': ML_unit_inds, 'manual_trial_selection': trial_inds, 'degraded':False, 'spike_threshold':None, 'trial_threshold':0},
               {'bin_width':50, 'same_trial_dur':True, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'AM', 'manual_unit_selection': AM_unit_inds, 'manual_trial_selection': trial_inds, 'degraded':False, 'spike_threshold':None, 'trial_threshold':0}]

    
# task args
KFold = 2
dimvals = np.arange(1,4)
task_args = [{'dim_vals':dimvals, 'n_folds':KFold, 'stratified_KFold':True, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':2, 'rng_or_seed':0}},
             {'dim_vals':dimvals, 'n_folds':KFold, 'stratified_KFold':True, 'dimreduc_method':'PCA', 'dimreduc_args': {}}]

