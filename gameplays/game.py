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
import tensorflow as tf

class Game(object):
    def __init__(self,white,black,verbose=True):
        self.white = white
        self.black = black
        self.verbose = verbose
        self.gamestate = gameplay.GameState()
    
    def play_till_end(self):
        winner = 'peace'
        moves  = []
        peace_round = 0
        remain_piece = gameplay.countpiece(self.gamestate.statestr)
        while True:
            start_time = time.time()
            if self.gamestate.move_number % 2 == 0:
                player_name = 'w'
                player = self.white
            else:
                player_name = 'b'
                player = self.black
            
            move,score = player.make_move(self.gamestate)
            if move is None:
                winner = 'b' if player_name == 'w' else 'w'
                break
            moves.append(move)
            if self.verbose:
                total_time = time.time() - start_time
                print('move {} {} play {} score {} use {:.2f}s pr {}'.format(self.gamestate.move_number,player_name,move,score if player_name == 'w' else -score,total_time,peace_round))
            game_end,winner_p = self.gamestate.game_end()
            if game_end:
                winner = winner_p
                break
            
            remain_piece_round = gameplay.countpiece(self.gamestate.statestr)
            if remain_piece_round < remain_piece:
                remain_piece = remain_piece_round
                peace_round = 0
            else:
                peace_round += 1
            if peace_round > conf.non_cap_draw_round:
                winner = 'peace'
                break
        print('winner: {}'.format(winner))
        return winner,moves
    
class NetworkPlayGame(Game):
    def __init__(self,network_w, network_b,**xargs):
        whiteplayer = players.NetworkPlayer('w',network_w,**xargs)
        blackplayer = players.NetworkPlayer('b',network_b,**xargs)
        super(NetworkPlayGame, self).__init__(whiteplayer,blackplayer)
        
class ContinusNetworkPlayGames(object):
    def __init__(self,network_w=None,network_b=None,white_name='net',black_name='net',random_switch=True,recoard_game=True,recoard_dir='data/distributed/',play_times=np.inf,distributed_server=None,distributed_dir='data/download_weight',**xargs):
        self.network_w = network_w
        self.network_b = network_b
        self.white_name = white_name
        self.black_name = black_name
        self.random_switch = random_switch
        self.play_times = play_times
        self.recoard_game = recoard_game
        self.recoard_dir = recoard_dir
        self.xargs = xargs
        self.distributed_server = distributed_server
        self.distributed_dir = distributed_dir
        self.nm = net_maintainer.NetMatainer(server=distributed_server,netdir=distributed_dir)
    
    def begin_of_game(self):
        pass
    
    def end_of_game(self,cbf_name,moves,cbfile):
        pass
    
    def play(self):
        num = 0
        while num < self.play_times:
            num += 1
            self.begin_of_game()
            if self.random_switch and random.random() < 0.5:
                self.network_w,self.network_b = self.network_b,self.network_w
                self.white_name,self.black_name = self.black_name,self.white_name
                
            network_play_game = NetworkPlayGame(self.network_w,self.network_b,**self.xargs)
            winner,moves = network_play_game.play_till_end()
            
            stamp = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
            date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
            cbfile = cbf.CBF(black=self.black_name,red=self.white_name,date=date,
                    site='北京',name='noname',datemodify=date,
                    redteam=self.white_name,blackteam=self.black_name,round='第一轮')
            cbfile.receive_moves(moves)
            
            randstamp = random.randint(0,1000)
            cbffilename = '{}_{}_mcts-mcts_{}-{}_{}.cbf'.format(stamp,randstamp,self.white_name,self.black_name,winner)
            cbf_name = os.path.join(self.recoard_dir,cbffilename)
            cbfile.dump(cbf_name)
            self.end_of_game(cbffilename,moves,cbfile)
            
class DistributedSelfPlayGames(ContinusNetworkPlayGames):
    def __init__(self,gpu_num=0,auto_update=True,**kwargs):
        self.gpu_num = gpu_num
        self.auto_update = auto_update
        super(DistributedSelfPlayGames, self).__init__(**kwargs)
    def begin_of_game(self):
        if not self.auto_update:
            print('no auto update')
            return
        latest_model_name = self.nm.get_update()
        model_dir = os.path.join(self.distributed_dir,latest_model_name)
        if self.network_w is None or self.network_b is None:
            print('initizing params {}'.format(latest_model_name))
            network = resnet.get_model(model_dir,
                common.board.create_uci_labels(),GPU_CORE=[self.gpu_num],
                FILTERS=conf.network_filters,NUM_RES_LAYERS=conf.network_layers)
            self.network_w = network
            self.network_b = network
            self.nm.updated(latest_model_name)
        else:
            if latest_model_name != self.nm.netname:
                (sess,graph),((X,training),(net_softmax,value_head)) = self.network_w
                with graph.as_default():
                    saver = tf.train.Saver(var_list=tf.global_variables())
                    saver.restore(sess,model_dir)
                self.nm.updated(latest_model_name)
            print('param updated {}'.format(model_dir))
        print("current weight {}".format(self.nm.netname))
    def end_of_game(self,cbf_name,moves,cbfile):
        if not conf.upload_gameplay_to_server:
            print("not sending gameplay to server according to config")
            return
        print("sending gameplay to server")
        data = urllib.parse.urlencode({'name':cbf_name,'content':cbfile.text})
        data = data.encode('utf-8')
        request = urllib.request.Request("{}/submit_chess".format(self.distributed_server))
        request.add_header("Content-Type","application/x-www-form-urlencoded;charset=utf-8")
        f = urllib.request.urlopen(request, data)
        print(f.read().decode('utf-8'))
        
class ValidationGames(ContinusNetworkPlayGames):
    pass
        
if __name__ == "__main__":
    #network = resnet.get_model(os.path.join(project_basedir,'data/prepare_weight/2018-06-07_14-13-24'),common.board.create_uci_labels(),GPU_CORE=[""],FILTERS=128,NUM_RES_LAYERS=7)
    #network_play_game = NetworkPlayGame(network,network,n_playout=400)
    #print(network_play_game.play_till_end())
    
    #cn = DistributedSelfPlayGames(distributed_server='http://10.109.247.219:10087',play_times=10,n_playout=400)
    #cn.play()
    
    network_a = resnet.get_model(os.path.join(project_basedir,'data/prepare_weight/2018-06-20_09-00-14'),common.board.create_uci_labels(),GPU_CORE=[""],FILTERS=128,NUM_RES_LAYERS=7)
    #network_a = resnet.get_model(os.path.join(project_basedir,'data/download_weight/2018-06-10_14-13-23'),common.board.create_uci_labels(),GPU_CORE=[""],FILTERS=128,NUM_RES_LAYERS=7)
    network_b = resnet.get_model(os.path.join(project_basedir,'data/models/update_model_model_2'),common.board.create_uci_labels(),GPU_CORE=[""],FILTERS=128,NUM_RES_LAYERS=7)
    
    vg = ValidationGames(network_w=network_a,network_b=network_b,white_name='oldnet',black_name='newnet',play_times=200,recoard_dir='data/validate',n_playout=400)
    vg.play()
    #cn = DistributedSelfPlayGames(network_w=network_a,network_b=network_a,distributed_server='http://10.109.247.219:10087',play_times=40,n_playout=400,auto_update=False,recoard_dir='data/validate')
    #cn.play()
    
    
