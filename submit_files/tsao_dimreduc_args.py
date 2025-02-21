import glob
import numpy as np
import pickle
 
# type
analysis_type = 'dimreduc'

# Paths
file_name = 'FOB_AnalysisDegraded_230809_140453_Alfie' # optional, for selecting individual sessions
data_path = '/home/marcush/Data/TsaoLabData/split/FOB'  
script_path = '/home/marcush/projects/neural_control/batch_analysis.py'
desc = 'testing'


#data_files = glob.glob('%s/degraded_only*.pickle' % data_path)
data_files = glob.glob(f'{data_path}/{file_name}.pickle')

#file_name_manual_units = file_name
#file_name_manual_units = 'degraded_only_v4_AnalysisDegraded_230322_214006_Jamie'
file_name_manual_units = 'degraded_only_v5_AnalysisDegraded_230809_140453_Alfie'
manual_keep_inds_path = "/home/marcush/Data/TsaoLabData/split/degraded/keepInds"

# In order to manually select units ::
with open(f'{manual_keep_inds_path}/{file_name_manual_units}_keep_inds_ML.pkl', 'rb') as f:
    ML_inds = pickle.load(f)
with open(f'{manual_keep_inds_path}/{file_name_manual_units}_keep_inds_AM.pkl', 'rb') as f:
    AM_inds = pickle.load(f)

# loader args
loader = 'tsao'
loader_args = [{'bin_width':25, 'same_trial_dur':True, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'ML', 'manual_unit_selection': ML_inds, 'degraded':False, 'spike_threshold':None, 'trial_threshold':0},
               {'bin_width':25, 'same_trial_dur':True, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'AM', 'manual_unit_selection': AM_inds, 'degraded':False, 'spike_threshold':None, 'trial_threshold':0}]


    
# task args
KFold = 5
dimvals = np.arange(1,60)
task_args = [{'dim_vals':dimvals, 'n_folds':KFold, 'stratified_KFold':True, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':5, 'rng_or_seed':0}},
             {'dim_vals':dimvals, 'n_folds':KFold, 'stratified_KFold':True, 'dimreduc_method':'PCA', 'dimreduc_args': {}}]

#task_args = [{'dim_vals':dimvals, 'n_folds':KFold, 'stratified_KFold':True, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':5, 'rng_or_seed':0, 'marginal_only':True}},
#            {'dim_vals':dimvals, 'n_folds':KFold, 'stratified_KFold':True, 'dimreduc_method':'PCA', 'dimreduc_args': {'marginal_only':True}}]


""""
Previous runs:

------------------------------------------------------------------------------------------
# type
analysis_type = 'dimreduc'

# Paths
script_path = '/home/marcush/projects/neural_control/batch_analysis.py'
desc = 'Performing dimreduc on degraded faces Tsao data (full params).'
data_path = '/home/marcush/Data/TsaoLabData/split/degraded'  
data_files = glob.glob('%s/*.pickle' % data_path)

# loader args
loader = 'tsao'
loader_args = [{'bin_width':25, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'ML'},
               {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'ML'},
               {'bin_width':25, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'AM'},
               {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'AM'}]

# This finds the minimal number of neurons (dimensions) across all recordings we're batch analyzing
maxUnits = np.inf
for data_file in data_files:
    with open(data_file, 'rb') as f:
        pat = pickle.load(f) 

    num_ML_units = len(  pat['regionIDs'][pat['regionIDs'] == pat['regions'].index('ML')] )
    num_AM_units = len(  pat['regionIDs'][pat['regionIDs'] == pat['regions'].index('AM')] )
    maxUnits = np.min(maxUnits, num_ML_units, num_AM_units)
    
# task args
KFold = 5
skipDims = 10 
dimvals = np.arange(1, maxUnits-1, skipDims)
task_args = [{'dim_vals':dimvals, 'n_folds':KFold, 'stratified_KFold':True, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'n_init':5, 'rng_or_seed':0}},
             {'dim_vals':dimvals, 'n_folds':KFold, 'stratified_KFold':True, 'dimreduc_method':'PCA', 'dimreduc_args': {}}]
------------------------------------------------------------------------------------------
script_path = '/home/marcush/projects/neural_control/batch_analysis.py'
desc = 'Performing dimreduc on degraded faces Tsao data'
data_path = '/home/marcush/Data/TsaoLabData/split/degraded'  
data_files = glob.glob('%s/*.pickle' % data_path)
loader = 'tsao'
analysis_type = 'dimreduc'
loader_args = [{'bin_width':25, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'ML'},
               {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'ML'},
               {'bin_width':25, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'AM'},
               {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'AM'}]
dimvals = np.arange(1, 60)
task_args = [{'dim_vals':dimvals, 'n_folds':5, 'stratified_KFold':True, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'n_init':5, 'rng_or_seed':0}},
             {'dim_vals':dimvals, 'n_folds':5, 'stratified_KFold':True, 'dimreduc_method':'PCA', 'dimreduc_args': {}}]

------------------------------------------------------------------------------------------
script_path = '/home/marcush/projects/neural_control/batch_analysis.py'
desc = 'Performing dimreduc on degraded faces Tsao data'
data_path = '/home/marcush/Data/TsaoLabData/split/degraded'  
data_files = glob.glob('%s/*.pickle' % data_path)
loader = 'tsao'
analysis_type = 'dimreduc'
loader_args = [{'bin_width':25, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'ML'},
               {'bin_width':50, 'filter_fn':'none', 'filter_kwargs':{}, 'boxcox':0.5, 'region':'AM'}]

#*** n_init and n_folds to larger vals *****
dimvals = np.arange(40, 45)
task_args = [{'dim_vals':dimvals, 'n_folds':2, 'stratified_KFold':True, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'n_init':2, 'rng_or_seed':0}},
             {'dim_vals':dimvals, 'n_folds':2, 'stratified_KFold':True, 'dimreduc_method':'PCA', 'dimreduc_args': {}}]
------------------------------------------------------------------------------------------
"""