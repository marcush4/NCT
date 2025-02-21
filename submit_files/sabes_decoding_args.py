import glob
import os
import numpy as np
import itertools

#script_path = '/global/homes/a/akumar25/repos/neural_control/batch_analysis.py'
#script_path = '/home/akumar/nse/neural_control/batch_analysis.py'
script_path = '/home/ankit_kumar/neural_control/batch_analysis.py'

#desc = 'Decode from indy norm dimreduc 2 (erroneously named indy norm decoding 2'
desc = 'Decoding of sabes trialized'
#data_path = '/mnt/Secondary/data/sabes'
#data_path = '/global/cscratch1/sd/akumar25/sabes'
data_path = '/home/ankit_kumar/Data/sabes'

# Data files specified by dimreduc files
data_files = ['']

# Load the data files and determine how many dof (neurons) there are in each recording
# data_files = ['%s/%s' % (data_path, data_file) for data_file in data_files]

loader = 'sabes_trialized'
analysis_type = 'decoding'

# Loader args are taken from the dimreduc files
loader_args = [[]]

# Grab all the dimreduc files. 
#dimreduc_files = glob.glob('/mnt/Secondary/data/sabes_M1subtrunc/sabes_M1subtrunc_*.dat')
dimreduc_files = glob.glob('/home/ankit_kumar/Data/FCCA_revisions/sabes_trialized_dimreduc/sabes_trialized_dimreduc*.dat')

#dimreduc_files = glob.glob(os.environ['SCRATCH'] + '/indy_dimreduc_parametric/indy_dimreduc_parametric_*.dat')

# each set of decoder_args. The rest of the iterables are handled
# in parallel at execution

# decoders = [{'method': 'svm', 'args':{'trainlag': 4, 'testlag': 4, 'decoding_window':5}}]
decoders = [{'method': 'lr', 'args':{'trainlag': 4, 'testlag': 4, 'decoding_window':5}}]

task_args = []
for param_comb in itertools.product(dimreduc_files, decoders):
	task_args.append({'dimreduc_file':param_comb[0],
					  'decoder':param_comb[1]})
