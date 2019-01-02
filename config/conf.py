import os
currentpath = os.path.dirname(os.path.realpath(__file__))
project_basedir = os.path.join(currentpath,'..')

distributed_datadir = os.path.join(project_basedir,'data/distributed')
distributed_server_weight_dir = os.path.join(project_basedir,'data/prepare_weight')
download_weight_dir = os.path.join(project_basedir,'data/download_weight')

validate_dir = os.path.join(project_basedir,'data/validate')
daily_log_dir = os.path.join(project_basedir,'data/log_update')
history_selfplay_dir = os.path.join(project_basedir,'data/history_selfplays')
model_dir = os.path.join(project_basedir,'data/models')
imsa_cbf = os.path.join(project_basedir,'data/imsa-cbf')

port = 10088
server = 'http://10.109.247.219'

train_lr = 0.01
train_epoch = 1
batch_size = 256
noup_flag = 'noup'
c_puct = 1.5

non_cap_draw_round = 120
sl_epoch = 30
sl_decay_eopch = 20

network_filters = 128
network_layers = 7
validate_gameplay = 200

train_playout = 800
val_playout = 800

train_temp_round = 3000
val_temp_round = 12

upload_gameplay_to_server = True
verbose = False
weight_up_immediately = True

