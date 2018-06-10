import os
import sys
currentpath = os.path.dirname(os.path.realpath(__file__))
project_basedir = os.path.join(currentpath,'..')
sys.path.append(project_basedir)
from gameplays import players,gameplay
from net import resnet
import common
import time

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
                print('move {} white {} score {} use {:.2f}s pr {}'.format(self.gamestate.move_number,move,score if player_name == 'w' else -score,total_time,peace_round))
            game_end,winner_p = self.gamestate.game_end()
            if game_end:
                winner = winner_p
                break
            
            # TODO remove 60 turn peace and add notact long catch, add no attack peace
            remain_piece_round = gameplay.countpiece(self.gamestate.statestr)
            if remain_piece_round < remain_piece:
                remain_piece = remain_piece_round
                peace_round = 0
            else:
                peace_round += 1
            if peace_round > 60:
                winner = 'peace'
                break
        print('winner: {}'.format(winner))
        return winner,moves
    
class NetworkPlayGame(Game):
    def __init__(self,network_w, network_b,**xargs):
        whiteplayer = players.NetworkPlayer('w',network_w,**xargs)
        blackplayer = players.NetworkPlayer('b',network_b,**xargs)
        super(NetworkPlayGame, self).__init__(whiteplayer,blackplayer)
        
if __name__ == "__main__":
    network = resnet.get_model(os.path.join(project_basedir,'data/prepare_weight/2018-06-07_14-13-24'),common.board.create_uci_labels(),GPU_CORE=[""],FILTERS=128,NUM_RES_LAYERS=7)
    network_play_game = NetworkPlayGame(network,network,n_playout=400)
    print(network_play_game.play_till_end())