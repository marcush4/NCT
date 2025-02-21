import glob
import sys; sys.path.append('/home/marcush/projects/neural_control')
from consolidation import consolidate_decoding, consolidate_dimreduc


dimreduc_file_path = '/home/marcush/Data/FrankLabData/neural_control_output/dimreduc_fullarg_frank_lab'
output_path = '/home/marcush/Data/FrankLabData/neural_control_output/dimreduc_fullarg_frank_lab/dimreduc_fullarg_frank_lab_glom.pickle'

#consolidate_dimreduc(dimreduc_file_path, output_path)


decode_file_path = '/home/marcush/Data/FrankLabData/neural_control_output/decoding_fullarg_frank_lab'
output_path = '/home/marcush/Data/FrankLabData/neural_control_output/decoding_fullarg_frank_lab/decoding_fullarg_frank_lab_glom.pickle'

consolidate_decoding(decode_file_path, output_path)
