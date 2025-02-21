import glob
import os
import numpy as np

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECK MARGINALS FLAG BELOW !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

script_path = '/home/marcush/projects/neural_control/batch_analysis.py'
desc = '[MARGINALS] Allen Visual Coding Dimreduc on sessions with >50 units for area VISp'
data_path = '/home/marcush/Data/AllenData' #'/clusterfs/NSDS_data/FCCA/data'
loader = 'AllenVC'
analysis_type = 'dimreduc'
Region = 'VISp'  
# Region Options: ['VISpm', 'LP', 'TH', 'DG', 'CA1', 'CA3', 'CA2', 'VISl', 'ZI', 'LGv', 'VISal', 'APN', 'POL', 'VISrl', 'VISam', 'LGd', 'ProS', 'SUB', 'VISp']

# VISp sessions with greater than 50 units: [732592105, 754312389, 798911424, 791319847, 754829445, 760693773, 757216464, 797828357, 762120172, 757970808, 799864342, 762602078, 755434585, 763673393, 760345702, 750332458, 715093703, 759883607, 719161530, 750749662, 756029989]
# Top 5 sessions with highest unit yield in CA1: [715093703, 761418226, 754312389, 798911424, 763673393]

# For all sessions:  #session_IDs = [ entry.name.split("_")[1] for entry in os.scandir(data_path) if entry.is_dir() and entry.name.startswith("session_") ]
session_IDs = [732592105, 754312389, 798911424, 791319847, 754829445, 760693773, 757216464, 797828357, 762120172, 757970808, 799864342, 762602078, 755434585, 763673393, 760345702, 750332458, 715093703, 759883607, 719161530, 750749662, 756029989]
data_files = [os.path.join(data_path, f"session_{session_ID}", f"session_{session_ID}.nwb") for session_ID in session_IDs]



# Each of these can be made into a list whose outer product is taken
loader_args = [{'region': Region, 'bin_width':25, 'preTrialWindowMS':50, 'postTrialWindowMS':100, 'boxcox':0.5},
               {'region': Region, 'bin_width':15, 'preTrialWindowMS':0, 'postTrialWindowMS':0, 'boxcox':0.5}]



KFold = 5
dimvals = np.arange(1,48)
task_args = [{'dim_vals':dimvals, 'n_folds':KFold, 'dimreduc_method':'LQGCA', 'dimreduc_args': {'T':3, 'loss_type':'trace', 'n_init':5, 'rng_or_seed':0, 'marginal_only':True}},
             {'dim_vals':dimvals, 'n_folds':KFold, 'dimreduc_method':'PCA', 'dimreduc_args': {'marginal_only':True}}]


