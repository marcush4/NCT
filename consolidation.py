import os
import pickle
import pandas as pd
import glob
import pdb
import numpy as np
from deepdiff import DeepDiff

import sys

sys.path.append('/home/akumar/nse/neural_control')
from loaders import load_sabes
from utils import apply_df_filters

def get_todo(path):

    # Enumerate args
    argfiles = glob.glob('%s/arg*.dat' % path)
    print(len(argfiles))
    todo = []
    for argfile in argfiles:
        # Open up the arg files
        with open(argfile, 'rb') as f:
            args = pickle.load(f)

        rfile = args['results_file']
        if not os.path.exists(rfile):
            todo.append(rfile)

    return todo

# Consolidate decoding results. Keep track of the directory where results are from and the original 
# replacement path: Use this if we should look for the results or dimreduc file in a different location than the argfile indicates
# This happens when we run a job on NERSC and consolidate it locally
def consolidate_decoding(src_path, save_path, replacement_drpath=None):

    argfiles = glob.glob('%s/arg*.dat' % src_path)
    result_list = []
    for argfile in argfiles:

        # Open up the arg files
        with open(argfile, 'rb') as f:
            args = pickle.load(f)

        rfile = args['results_file']
        rfile = rfile.split('/')[-1]
        rfile = src_path + '/' + rfile

        if os.path.exists(rfile):
            with open(rfile, 'rb') as f:
                result = pickle.load(f)


            # Could be the case (e.g. in rand decoding) that result_
            # is itself a list of dictionaries
            if isinstance(result[0], list):
                # Flatten
                result = [r for result_ in result for r in result_]
            for result_ in result:
                for k, v in args.items():
                    if k == 'loader_args':
                        result_['dec_loader_args'] = v
                    if type(k) == dict:
                        for k_, v_ in k.items():
                            result_[k_] = v_
                    else:
                        try:
                            result_[k] = v
                        except:
                            pdb.set_trace()
            # Need to grab some information from the dimreduc arg file
            if replacement_drpath is None:  
                dimreduc_path = '/'.join(args['task_args']['dimreduc_file'].split('/')[:-1])
            else:
                dimreduc_path = replacement_drpath

            dimreduc_no = args['task_args']['dimreduc_file'].split('_')[-1].split('.dat')[0]
            dimreduc_argfile = '%s/arg%s.dat' % (dimreduc_path, dimreduc_no)
            with open(dimreduc_argfile, 'rb') as f:
                dr_args = pickle.load(f)

            for result_ in result:
                for k, v in dr_args.items():
                    # Use the loader args from the decoding arg files
                    # This is used, for example, when we smooth at decoding
                    # but not at dimreduc
                    if type(k) == dict:
                        for k_, v_ in k.items():
                            result_[k_] = v_
                    else:
                        result_[k] = v

            result_list.extend(result)
        else:
            continue

    # For ease of use, use data file names that do not involve the directory path
    for r in result_list:
        r['data_file'] = r['data_file'].split('/')[-1]

    try:
        # Try using pickle to save the data
        with open(save_path, 'wb') as f:
            f.write(pickle.dumps(result_list))
            f.write(pickle.dumps(src_path))
        print("Data saved using pickle.")
        
    except MemoryError:
        print("MemoryError occurred. Using joblib to save the data.")
        from joblib import dump, load
        
        # Save the data using joblib, then load it so it can be saved as a pickle.
        # Unsure if this is necessary, or if I can immedietly convert result_list to DF then save with pickle...
        dump(result_list, save_path)
        data = load(save_path)  
        df_decode = pd.DataFrame(data)
        with open(save_path, 'wb') as f:
            pickle.dump(df_decode, f)

        print("Data saved using joblib (then converted to regular .pkl).")

 
def consolidate_dimreduc(src_path, save_path):
    
    argfiles = glob.glob('%s/arg*.dat' % src_path)
    print(len(argfiles))
    result_list = []
    for argfile in argfiles:

        # Open up the arg files
        with open(argfile, 'rb') as f:
            args = pickle.load(f)

        rfile = args['results_file']
        with open(rfile, 'rb') as f:
            result = pickle.load(f)

        for result_ in result:
            for k, v in args.items():
                if type(k) == dict:
                    for k_, v_ in k.items():
                        result_[k_] = v_
                else:
                    result_[k] = v

        result_list.extend(result)

    # For ease of use, use data file names that do not involve the directory path
    for r in result_list:
        r['data_file'] = r['data_file'].split('/')[-1]
    
    # Save the result list, directory path
    with open(save_path, 'wb') as f:
        f.write(pickle.dumps(result_list))
        f.write(pickle.dumps(src_path))
