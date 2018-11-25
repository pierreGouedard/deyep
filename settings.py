import os

deyep_home = os.path.join(os.path.expanduser("~"), 'deyep')
deyep_data_path = os.path.join(deyep_home, 'DATA/PROJECTS/{}')
deyep_params_path = os.path.join(deyep_data_path, 'params.yml')
deyep_raw_path = os.path.join(deyep_data_path, 'RAW')
deyep_io_path = os.path.join(deyep_data_path, 'FEATURES')
deyep_imputer_path = os.path.join(deyep_data_path, 'IMPUTERS')
deyep_network_path = os.path.join(deyep_data_path, 'NETWORKS')
deyep_prod_path = os.path.join(deyep_data_path, 'PROD')
deyep_driver_file_tmpdir = '/tmp'
