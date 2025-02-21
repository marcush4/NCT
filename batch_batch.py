from batch_util import init_batch
import subprocess

# submit_files = ['submit_files/sabes_dimreduc_args.py',
#                 'submit_files/sabes_dimreduc_args_marginal.py',
#                 'submit_files/sabes_dimreduc_argsS1.py',
#                 'submit_files/sabes_dimreduc_argsS1_marginal.py']

#submit_files = ['submit_files/sabes_decoding_args_subset.py',
#                'submit_files/sabes_decoding_args_subsetS1.py']
                # 'submit_files/sabes_decoding_args_marginal.py',
                # 'submit_files/sabes_decoding_argsS1_marginal.py']
submit_files = ['submit_files/peanut_decoding_args.py',
                'submit_files/mcmaze_decoding_args.py',
                'submit_files/sabes_decoding_args.py']

save_dirs = ['/home/ankit_kumar/Data/FCCA_revisions/peanut_marginal_decoding',
             '/home/ankit_kumar/Data/FCCA_revisions/mcmaze_marginal_decoding',
             '/home/ankit_kumar/Data/FCCA_revisions/sabes_trialized_decoding']
            #  '/mnt/Secondary/data/sabes_S1subtruncmarginal_dec']
            #  '/mnt/Secondary/data/sabes_M1subtruncmarginal_dec',

for i, submit_file in enumerate(submit_files):
    init_batch(submit_file, save_dirs[i], local=True)

# change permissions
for sd in save_dirs:
    subprocess.run(['chmod', '777 ' '%s/sbatch_resume.sh' % sd])

# write sbatch scripts to a new .sh file
# with open('run.sh', 'w') as f:
#     for sd in save_dirs:
#         f.write('%s/sbatch_resume.sh\n' % sd)

# subprocess.run(['chmod', '777 ', 'run.sh'])
