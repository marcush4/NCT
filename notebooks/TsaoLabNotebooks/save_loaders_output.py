import pickle
import pandas as pd
import sys; sys.path.append("/home/marcush/projects/neural_control")  # Allows access to all the scripts/modules in the larger directory
from loaders import load_tsao


path = '/home/marcush/Data/TsaoLabData/neural_control_output/degraded_small_batch/degraded_small_batch_glom.pickle'
with open(path, 'rb') as f:
    dat = pickle.load(f) 

df_dimreduc = pd.DataFrame(dat)


def make_hashable(d):
    """ Recursively convert a dictionary into a hashable type (tuples of tuples). """
    if isinstance(d, dict):
        return tuple((key, make_hashable(value)) for key, value in sorted(d.items()))
    elif isinstance(d, list):
        return tuple(make_hashable(value) for value in d)
    else:
        return d

# Assuming df_dimreduc['loader_args'] is your column with dictionaries
unique_hashes = set(make_hashable(d) for d in df_dimreduc['loader_args'])

# Convert each hashable entity back to a dictionary if necessary
unique_dicts = [dict(t) for t in unique_hashes]  # This step might need adjustment based on your data structure


data_path = df_dimreduc['data_path'][0] + '/' + df_dimreduc['data_file'][0]
output_dir = df_dimreduc['data_path'][0] + '/loader_data/'

for d in unique_dicts:
    bin_width = d['bin_width']
    boxcox = d['boxcox']
    filter_fn = d['filter_fn']
    filter_kwargs = d['filter_kwargs']
    region = d['region']

    save_name = f"{df_dimreduc['data_file'][0]}_{bin_width}_{region}.pickle"
    output_path = output_dir + save_name

    dat = load_tsao(data_path, bin_width=bin_width, region=region, boxcox=boxcox, filter_fn=filter_fn, filter_kwargs=filter_kwargs)

    with open(output_path, 'wb') as file:
        pickle.dump(dat, file)

   