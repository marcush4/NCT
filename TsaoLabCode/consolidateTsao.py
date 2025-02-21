import glob
import sys; sys.path.append('/home/marcush/projects/neural_control')
from consolidation import consolidate_decoding, consolidate_dimreduc


dimreduc_file_path = '/home/marcush/Data/OrganoidData/neural_control_output/organoids_dimreduc'
output_path = '/home/marcush/Data/OrganoidData/neural_control_output/organoids_dimreduc/organoids_dimreduc_glom.pickle'

consolidate_dimreduc(dimreduc_file_path, output_path)


#decode_file_path = '/home/marcush/Data/TsaoLabData/neural_control_output_new/decoding_degOnly_marginals_230322_214006_Jamie'
#output_path = '/home/marcush/Data/TsaoLabData/neural_control_output_new/decoding_degOnly_marginals_230322_214006_Jamie/decoding_degOnly_marginals_230322_214006_Jamie_glom.pickle'
#consolidate_decoding(decode_file_path, output_path)
 