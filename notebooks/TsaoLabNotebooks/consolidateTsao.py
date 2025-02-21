import glob
import sys
sys.path.append('/home/marcush/projects/neural_control/')
sys.path.append('/home/marcush/projects/neural_control/analysis_scripts/')
sys.path.append('/home/marcush/projects/github_repos')
from consolidation import consolidate_decoding, consolidate_dimreduc



#dimreduc_file_path = '/home/marcush/Data/FrankLabData/neural_control_output/dimreduc_fullarg_frank_lab_marginals'
#output_path = '/home/marcush/Data/FrankLabData/neural_control_output/dimreduc_fullarg_frank_lab_marginals/dimreduc_fullarg_frank_lab_marginals_glom.pickle'

#consolidate_dimreduc(dimreduc_file_path, output_path)



decode_file_path = '/home/marcush/Data/AllenData/neural_control_output/decoding_AllenVC_VISp_marginals'

output_path = '/home/marcush/Data/AllenData/neural_control_output/decoding_AllenVC_VISp_marginals/decoding_AllenVC_VISp_marginals_glom.pickle'

consolidate_decoding(decode_file_path, output_path)


