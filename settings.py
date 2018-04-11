import os

deyep_home = os.path.join(os.path.expanduser("~"), 'deyep')
deyep_data_path = os.path.join(deyep_home, 'DATA/PROJECTS/{}')
deyep_params_path = os.path.join(deyep_data_path, 'params.yml')
deyep_raw_path = os.path.join(deyep_data_path, 'RAW')
deyep_io_path = os.path.join(deyep_data_path, 'IO')
deyep_generator_path = os.path.join(deyep_data_path, 'GENERATORS')
deyep_network_path = os.path.join(deyep_data_path, 'NETWORKS')
deyep_driver_file_tmpdir = '/tmp'
