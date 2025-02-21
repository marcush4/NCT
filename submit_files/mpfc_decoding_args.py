import glob
import os
import numpy as np
import itertools

#script_path = '/global/homes/a/akumar25/repos/localization/batch_analysis_sabes.py'
#script_path = '/home/akumar/nse/neural_control/batch_analysis.py'
script_path = '/home/ankit_kumar/neural_control/batch_analysis.py'

desc = 'decode from mPFC dimreduc'
data_path = '/clusterfs/NSDS_data/franklabdata/dataset1'

data_files = [data_path]
loader = 'franklab_new'
analysis_type = 'decoding'

# Loader args are taken from the dimreduc files
loader_args = [[]]

# Grab all the dimreduc files. 
dimreduc_files = glob.glob('/home/ankit_kumar/Data/FCCA_revisions/mpFC_dimreduc/mpFC_dimreduc*.dat')

# Create separate set of task args for each dimreduc file and
# each set of decoder_args. The rest of the iterables are handled
# in parallel at execution
decoders = [{'method': 'lr', 'args':{'trainlag': 0, 'testlag': 0, 'decoding_window': 6}},
            {'method': 'lr', 'args':{'trainlag': 1, 'testlag': 1, 'decoding_window': 6}},
            {'method': 'lr', 'args':{'trainlag': -1, 'testlag': -1, 'decoding_window': 6}},
            {'method': 'lr', 'args':{'trainlag': 2, 'testlag': 2, 'decoding_window': 6}},
            {'method': 'lr', 'args':{'trainlag': -2, 'testlag': -2, 'decoding_window': 4}}]

task_args = []
for param_comb in itertools.product(dimreduc_files, decoders):
	task_args.append({'dimreduc_file':param_comb[0],
					  'decoder':param_comb[1]})
