import glob
import itertools
import os
 
# Path to raw data
data_path = '/home/marcush/Data/TsaoLabData/split/FOB' 
# Path to dimreduc files
dimreduc_files = glob.glob('/home/marcush/Data/TsaoLabData/neural_control_output_new/dimreduc_FOB_230322_214006_Jamie/*.dat')

script_path = '/home/marcush/projects/neural_control/batch_analysis.py'
desc = "Alfie  Final Analyses"
 
# Data files and loader_args specified by dimreduc files
data_files = [' ']
loader_args = [[]]

loader = 'tsao'
analysis_type = 'decoding'

# Grab all the dimreduc files (ignore arg files)
dimreduc_files = [file for file in dimreduc_files if not os.path.basename(file).startswith('arg')]

decoders = [{'method': 'logreg', 'args':{}}]

task_args = []
for param_comb in itertools.product(dimreduc_files, decoders):
	task_args.append({'dimreduc_file':param_comb[0],
					  'decoder':param_comb[1]})

""""
Previous runs:
------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------
"""