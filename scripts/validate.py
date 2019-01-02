import os
import sys
currentpath = os.path.dirname(os.path.realpath(__file__))
project_basedir = os.path.join(currentpath,'..')
sys.path.append(project_basedir)
from gameplays import players,gameplay
from net import resnet
import common
import time
import numpy as np
import random
from cchess_zero import cbf
from config import conf
import os
from net import net_maintainer
import urllib
from gameplays.game import ValidationGames
import argparse

parser = argparse.ArgumentParser(description="mcts self play script") 
parser.add_argument('--verbose', '-v', action='store_true', help='verbose mode')
parser.add_argument('--gpu', '-g' , choices=[int(i) for i in list(range(8))],type=int,help="gpu core number",default=0)
parser.add_argument('--server', '-s' ,type=str,help="distributed server location",default=conf.server)
args = parser.parse_args()

gpu_num = int(args.gpu)
server = args.server
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)


new_name = sorted([i[:-6] for i in os.listdir(conf.distributed_server_weight_dir) if '.index' in i])[-1]
if conf.weight_up_immediately:
    old_name = sorted([i[:-6] for i in os.listdir(conf.distributed_server_weight_dir) if '.index' in i and conf.noup_flag not in i])[-2]
else:
    old_name = sorted([i[:-6] for i in os.listdir(conf.distributed_server_weight_dir) if '.index' in i and conf.noup_flag not in i])[-1]

print("------------------------------------------")
print("loading new model {}".format(new_name))
print("loading old model {}".format(old_name))
print("------------------------------------------")
labels = common.board.create_uci_labels()
netold = resnet.get_model('{}/{}'.format(conf.distributed_server_weight_dir,old_name),labels,GPU_CORE=[gpu_num],FILTERS=conf.network_filters,NUM_RES_LAYERS=conf.network_layers)
netnew = resnet.get_model('{}/{}'.format(conf.distributed_server_weight_dir,new_name),labels,GPU_CORE=[gpu_num],FILTERS=conf.network_filters,NUM_RES_LAYERS=conf.network_layers)
recoard_dir = '{}/{}'.format(conf.validate_dir,new_name)
if not os.path.exists(recoard_dir):
    print("creating dir {}".format(recoard_dir))
    os.mkdir(recoard_dir)
else:
    print("dir already exist")

number_played = len(os.listdir(recoard_dir))
while number_played < conf.validate_gameplay:
    vg = ValidationGames(network_w=netold,network_b=netnew,white_name='oldnet',black_name='newnet',play_times=1,recoard_dir=recoard_dir,n_playout=conf.val_playout,temp_round=conf.val_temp_round,c_puct=conf.c_puct)
    vg.play()
    number_played = len(os.listdir(recoard_dir))
    print("played {} th game".format(number_played))
