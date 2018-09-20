import tornado.ioloop
import shutil
import tornado.web
import argparse
import os
import sys
import numpy as np

currentpath = os.path.dirname(os.path.realpath(__file__))
project_basedir = os.path.join(currentpath,'..')
sys.path.append(project_basedir)

from config import conf
datadir = conf.distributed_datadir


validate_dirs = os.listdir(conf.validate_dir)
validate_dirs = [i for i in validate_dirs if i != '_blank']
validate_dirs = sorted(validate_dirs)
validate_dirs = [os.path.join(conf.validate_dir,i) for i in validate_dirs]

def add_score(onedic,key,point):
    onedic.setdefault(key,0)
    onedic[key] += point
def cal_points(gameplays):
    point_dic = {}
    for onegame in gameplays:
        if onegame[-3:] != 'cbf':
            continue
        winner = onegame.split('_')[-1].split('.')[0]
        player1 = onegame.split('_')[-2].split('-')[0]
        player2 = onegame.split('_')[-2].split('-')[1]
        assert(winner in ['w','b','peace'])
        if winner == 'w':
            add_score(point_dic,player1,1)
            add_score(point_dic,player2,0)
        elif winner == 'b':
            add_score(point_dic,player1,0)
            add_score(point_dic,player2,1)
        elif winner == 'peace':
            add_score(point_dic,player1,0.5)
            add_score(point_dic,player2,0.5)
            add_score(point_dic,'peace',1)
        else:
            raise
    return point_dic

game_numbers = [0]
game_numbers_identity = [0]
elu_points = [0]
validate_games = [0]
win_rate = [0]
dates = ['start']
peace_rates = [0]
delta_elo = [0]

one_dir = validate_dirs[-1]
one_date = one_dir.split('/')[-1].replace("_{}".format(conf.noup_flag),"")
gameplays = os.listdir(one_dir)
pointcdic = cal_points(gameplays)
game_num = len(gameplays)

try:
    gn = len(os.listdir(os.path.join(conf.history_selfplay_dir,one_date)))
except:
    gn = 0
game_numbers.append(game_numbers[-1] + gn)
game_numbers_identity.append(gn)
old_score = pointcdic['oldnet'] / game_num
peace_rate = pointcdic.get('peace',0) / game_num
elo = np.log10(1 / old_score - 1) * 400

print(elo,one_date)

one_noupweight = os.path.join(conf.distributed_server_weight_dir,one_date)
one_noupweight_up = "{}".format(one_noupweight)
one_noupweight_noup = "{}_{}".format(one_noupweight,conf.noup_flag)
print(one_noupweight,one_noupweight_up,one_noupweight_noup)
print(one_noupweight_up)
if elo < -50:
    print("cannot up weight: win rate < 50%")
else:
    print(one_noupweight_noup)
    if os.path.exists(one_noupweight_noup + '.index'):
        print("up weight")
        for f in ['data-00000-of-00001','meta','index']:
            src = one_noupweight_noup + '.' + f
            dst = one_noupweight_up + '.' + f
            print("copying file from {} to {}".format(src,dst))
            shutil.copyfile(src,dst)
    else:
        print("No weight to up")
