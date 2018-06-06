import os
currentpath = os.path.dirname(os.path.realpath(__file__))
project_basedir = os.path.join(currentpath,'..')

distributed_datadir = os.path.join(project_basedir,'data/distributed')
distributed_server_weight_dir = os.path.join(project_basedir,'data/prepare_weight')

validate_dir = os.path.join(project_basedir,'data/validate')
daily_log_dir = os.path.join(project_basedir,'data/log_update')