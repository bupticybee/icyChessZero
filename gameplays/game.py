import os
import sys
currentpath = os.path.dirname(os.path.realpath(__file__))
project_basedir = os.path.join(currentpath,'..')
sys.path.append(project_basedir)
from gameplays import players,gameplay
from net import resnet
import common

class Game(object):
    def __init__(self,white,black,verbose=True):
        self.white = white
        self.black = black
        self.verbose = verbose
        self.gamestate = gameplay.GameState()
    
    def play_till_end(self):
        winner = 'peace'
        moves  = []
        while True:
            move,score = self.white.make_move(self.gamestate)
            if move is None:
                winner = 'b'
                break
            moves.append(move)
            if self.verbose:
                print('move {} white {} score {}'.format(self.gamestate.move_number,move,score))
            game_end,winner_p = self.gamestate.game_end()
            if game_end:
                winner = winner_p
                break

            move,score = self.black.make_move(self.gamestate)
            if move is None:
                winner = 'w'
                break
            moves.append(move)
            if self.verbose:
                print('move {} black {} score {}'.format(self.gamestate.move_number,move,-score))
            game_end,winner_p = self.gamestate.game_end()
            if game_end:
                winner = winner_p
                break
            # TODO add 60 turn peace and check game_end logic
        print('winner: {}'.format(winner))
        return winner,moves
    
class NetworkPlayGame(Game):
    def __init__(self,network_w, network_b):
        whiteplayer = players.NetworkPlayer('w',network_w)
        blackplayer = players.NetworkPlayer('b',network_b)
        super(NetworkPlayGame, self).__init__(whiteplayer,blackplayer)
        
if __name__ == "__main__":
    network = resnet.get_model(os.path.join(project_basedir,'data/prepare_weight/2018-06-07_14-13-24'),common.board.create_uci_labels(),GPU_CORE=[""],FILTERS=128,NUM_RES_LAYERS=7)
    network_play_game = NetworkPlayGame(network,network)
    print(network_play_game.play_till_end())