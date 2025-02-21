import glob
import os
import numpy as np

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECK MARGINALS FLAG BELOW !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

script_path = '/home/marcush/projects/neural_control/batch_analysis.py'
desc = 'Testing brain organoid data'
data_path = '/home/marcush/Data/OrganoidData' #'/clusterfs/NSDS_data/FCCA/data'
loader = 'load_organoids'
analysis_type = 'dimreduc'

data_files = ["/home/marcush/Data/OrganoidData/AllSessions.pkl"]


# Each of these can be made into a list whose outer product is taken
loader_args = [{'trial_length':2000,'bin_width':100, 'boxcox':0.5},
               {'trial_length':1000,'bin_width':50, 'boxcox':0.5}]



KFold = 5
dimvals = np.arange(1,100)
task_args = [{'dim_vals':dimvals, 'n_folds':KFold, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':5, 'rng_or_seed':0}},
             {'dim_vals':dimvals, 'n_folds':KFold, 'dimreduc_method':'PCA', 'dimreduc_args': {}}]


