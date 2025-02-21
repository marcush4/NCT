import argparse
import glob
import mat73
import numpy as np
import os
import pickle

#Default lcoation of unsplit tsao data on the Bouchard lab server
unsplit_tsao_data_path = "/home/marcush/Data/TsaoLabData/unsplit"
split_tsao_data_path = "/home/marcush/Data/TsaoLabData/split"


# CHECK THAT LOADERS CAN CALL THESE FILES FIRST??

def split_data(paths):

    print("Begining data splits...")
    for experiment_path in paths:
        print(f"Begin loading: {experiment_path}")
        f = mat73.loadmat(experiment_path) 
        print(f"Done loading: {experiment_path}")
        sessionName = os.path.split(os.path.dirname(experiment_path))[-1]

        AllParadigms = f['session']['paradigm_names']
        non_empty_indices = [index for index, value in enumerate(AllParadigms) if value]

        j=1
        for valid_idx in non_empty_indices:
             print(f"Splitting paradigm {j} of {len(non_empty_indices)}")

             # Save all the data for this paradigm, later pre-processed
             new_data_dict = f['session']['paradigms_data'][valid_idx][0]

             # Save some other, useful info from the recording
             paradigmName = AllParadigms[valid_idx]
             new_data_dict["ParadigmName"] = paradigmName
             new_data_dict["regions"] = f['session']['regions']
             new_data_dict["regionIDs"] = f['session']['unit_probes_index'] - 1
             new_data_dict["probeID"] = f['session']['unit_probes_index'] - 1   
             new_data_dict["channelID"] = f['session']['unit_channel'] - 1
             new_data_dict["waveforms"] = f['session']['waveforms']['unit_waveform_uV']
             new_data_dict["waveforms_time"] = f['session']['waveforms']['unit_waveform_time']


             with open("{}/{}_{}.pickle".format(split_tsao_data_path, paradigmName, sessionName), 'wb') as handle:
                 pickle.dump(new_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
             j+=1


def main():
    parser = argparse.ArgumentParser(description="Process a file.")
    parser.add_argument('--file', type=str, help='Path to the file', default=None)
    args = parser.parse_args()
    
    if args.file:
        path = glob.glob(f'{args.file}/**/*.mat', recursive=True)
    else:
        path = glob.glob(f'{unsplit_tsao_data_path}/**/*.mat', recursive=True)

    split_data(path)

if __name__ == "__main__":
    main()


# pass function the filename of new tsao data and it will spit out the paradigms chooped up
# if None then will run on all unsplit data