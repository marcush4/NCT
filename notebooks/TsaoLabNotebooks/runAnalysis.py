import os
import sys; sys.path.append('/home/marcush/projects/neural_control/')
from batch_util import init_batch


#dimreduc_deg_230322_214006_Jamie
jobdir = '/home/marcush/Data/FrankLabData/neural_control_output/decoding_fullarg_frank_lab_marginals'
submit_file = '/home/marcush/projects/neural_control/submit_files/franklab_new_decoding_args.py'
# Initializes the paths/folders for analysis
init_batch(submit_file, jobdir, local=True)

# ************ Actually calls sbatch_resume (which calls batch_analysis) from the command line, with the appropriate conda env ************
#os.system(f"source /home/marcush/Data/anaconda3/etc/profile.d/conda.sh && conda activate ncontrol && chmod 777 {jobdir}/sbatch_resume.sh && {jobdir}/sbatch_resume.sh")


""""
Previous runs:
------------------------------------------------------------------------------------------

jobdir = '/home/marcush/Data/TsaoLabData/neural_control_output/degraded_dimreduc_med_batch'
submit_file = '/home/marcush/projects/neural_control/submit_files/tsao_dimreduc_args.py'
# Initializes the paths/folders for analysis
init_batch(submit_file, jobdir, local=True)
------------------------------------------------------------------------------------------
jobdir = '/home/marcush/Data/TsaoLabData/neural_control_output/degraded_decoding_small_batch'
submit_file = '/home/marcush/projects/neural_control/submit_files/tsao_decoding_args.py'
# Initializes the paths/folders for analysis
init_batch(submit_file, jobdir, local=True)

------------------------------------------------------------------------------------------
jobdir = '/home/marcush/Data/TsaoLabData/neural_control_output/degraded_small_batch'
submit_file = '/home/marcush/projects/neural_control/submit_files/tsao_dimreduc_args.py'
# Initializes the paths/folders for analysis
init_batch(submit_file, jobdir, local=True)
------------------------------------------------------------------------------------------

"""