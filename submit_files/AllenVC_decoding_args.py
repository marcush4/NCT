import glob
import itertools
import os


data_path = '/home/marcush/Data/AllenData' 
dimreduc_files = glob.glob('/home/marcush/Data/AllenData/neural_control_output/dimreduc_AllenVC_VISp_marginals/*.dat')
loader = 'AllenVC'

desc = "[MARGINALS] VISp Decoding"
analysis_type = 'decoding'
script_path = '/home/marcush/projects/neural_control/batch_analysis.py'


# Data files and loader_args specified by dimreduc files
data_files = [' ']
loader_args = [[]]


dimreduc_files = [file for file in dimreduc_files if not os.path.basename(file).startswith('arg')]
decoders = [{'method': 'logreg', 'args':{}}]

task_args = []
for param_comb in itertools.product(dimreduc_files, decoders):
	task_args.append({'dimreduc_file':param_comb[0],
					  'decoder':param_comb[1]})
