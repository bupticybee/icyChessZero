import os
import sys
currentpath = os.path.dirname(os.path.realpath(__file__))
project_basedir = os.path.join(currentpath,'..')
sys.path.append(project_basedir)
from config import conf
from gameplays.game import DistributedSelfPlayGames
import argparse

parser = argparse.ArgumentParser(description="mcts self play script") 
parser.add_argument('--gpu', '-g' , choices=[int(i) for i in list(range(8))],type=int,help="gpu core number",default=0)
args = parser.parse_args()
gpu_num = int(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

cn = DistributedSelfPlayGames(
    gpu_num = gpu_num,
    distributed_server='{}:{}'.format(conf.server,conf.port),
    n_playout=conf.train_playout,
    recoard_dir=conf.distributed_datadir,
    c_puct=conf.c_puct,
    distributed_dir=conf.download_weight_dir,
    dnoise=True,
    is_selfplay=True,
)
cn.play()
