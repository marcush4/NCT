import multiprocessing as mp
import subprocess

# Define the parameters for each run (remove runInd from here)
"""param_list = [
    {'bin_width': 25, 'spike_threshold': 0, 'speed_threshold': False, 'trialize': False, 'dimreduc_args_query': "{'T': 3, 'loss_type': 'trace', 'n_init': 10}", 'decoders_args_query': "{'trainlag': 1, 'testlag': 1, 'decoding_window': 6}"},
    {'bin_width': 25, 'spike_threshold': 0, 'speed_threshold': False, 'trialize': False, 'dimreduc_args_query': "{'T': 3, 'loss_type': 'trace', 'n_init': 10}", 'decoders_args_query': "{'trainlag': 0, 'testlag': 0, 'decoding_window': 6}"},
    {'bin_width': 25, 'spike_threshold': 0, 'speed_threshold': False, 'trialize': True, 'dimreduc_args_query': "{'T': 3, 'loss_type': 'trace', 'n_init': 10}", 'decoders_args_query': "{'trainlag': 1, 'testlag': 1, 'decoding_window': 6}"},
    {'bin_width': 25, 'spike_threshold': 100, 'speed_threshold': False, 'trialize': False, 'dimreduc_args_query': "{'T': 3, 'loss_type': 'trace', 'n_init': 10}", 'decoders_args_query': "{'trainlag': 1, 'testlag': 1, 'decoding_window': 6}"},
    {'bin_width': 25, 'spike_threshold': 100, 'speed_threshold': False, 'trialize': True, 'dimreduc_args_query': "{'T': 3, 'loss_type': 'trace', 'n_init': 10}", 'decoders_args_query': "{'trainlag': 1, 'testlag': 1, 'decoding_window': 6}"},

]"""
param_list = [
    {'bin_width': 25, 'spike_threshold': 0, 'speed_threshold': False, 'trialize': False, 'dimreduc_args_query': "{'T': 3, 'loss_type': 'trace', 'n_init': 10}", 'decoders_args_query': "{'trainlag': 1, 'testlag': 1, 'decoding_window': 6}"}
]


def run_script(params):
    idx, param_set = params
    cmd = [
        'python', '/home/marcush/projects/neural_control/analysis_scripts/turnkey/activity_regression_script.py', 
        '--bin_width', str(param_set['bin_width']),
        '--spike_threshold', str(param_set['spike_threshold']),
        '--speed_threshold', str(param_set['speed_threshold']),
        '--trialize', str(param_set['trialize']),
        '--dimreduc_args_query', param_set['dimreduc_args_query'],
        '--decoders_args_query', param_set['decoders_args_query'],
        '--runInd', str(idx)  # Automatically assign runInd based on the index
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    # Enumerate the param_list so each has a unique index (runInd)
    param_list_with_index = list(enumerate(param_list))
    
    # Use all available cores
    pool = mp.Pool(mp.cpu_count())
    pool.map(run_script, param_list_with_index)
    pool.close()
    pool.join()