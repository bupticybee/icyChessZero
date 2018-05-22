import xmltodict
from cchess import *
from cchess_zero.gameboard import *
pieceset = {'A',
 'B',
 'C',
 'K',
 'N',
 'P',
 'R',
 'a',
 'b',
 'c',
 'k',
 'n',
 'p',
 'r'}

countpiece = lambda x: sum([1 for i in x if i in pieceset])

class GameState():
    def __init__(self):
        self.statestr = 'RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr'
        self.currentplayer = 'w'
        self.ys = '9876543210'[::-1]
        self.xs = 'abcdefghi'
        self.pastdic = {}
        self.maxrepeat = 0
        self.lastmove = ""
        
    def get_king_pos(self):
        board = self.statestr.replace("1", " ")
        board = board.replace("2", "  ")
        board = board.replace("3", "   ")
        board = board.replace("4", "    ")
        board = board.replace("5", "     ")
        board = board.replace("6", "      ")
        board = board.replace("7", "       ")
        board = board.replace("8", "        ")
        board = board.replace("9", "         ")
        board = board.split('/')

        for i in range(3):
            pos = board[i].find('K')
            if pos != -1:
                K = "{}{}".format(self.xs[pos],self.ys[i])
        for i in range(-1,-4,-1):
            pos = board[i].find('k')
            if pos != -1:
                k = "{}{}".format(self.xs[pos],self.ys[i])
        return K,k
            
    def game_end(self):
        #if self.statestr.find('k') == -1:
        #    return True,'w'
        #elif self.statestr.find('K') == -1:
        #    return True,'b'
        wk,bk = self.get_king_pos()
        if self.maxrepeat >= 3 and (self.lastmove[-2:] != wk and self.lastmove[-2:] != bk):
            return True,self.get_current_player()
        targetkingdic = {'b':wk,'w':bk}
        moveset = GameBoard.get_legal_moves(self.statestr,self.get_current_player())
        
        targetset = set([i[-2:] for i in moveset])
        
        targ_king = targetkingdic[self.currentplayer]
        if targ_king in targetset:
            return True,self.currentplayer
        return False,None
    
    def get_current_player(self):
        return self.currentplayer
    
    def do_move(self,move):
        self.lastmove = move
        self.statestr = GameBoard.sim_do_action(move,self.statestr)
        if self.currentplayer == 'w':
            self.currentplayer = 'b'
        elif self.currentplayer == 'b':
            self.currentplayer = 'w'
        self.pastdic.setdefault(self.statestr,0)
        self.pastdic[self.statestr] += 1
        self.maxrepeat = max(self.maxrepeat,self.pastdic[self.statestr])

class GamePlay:
    def __init__(self):
        self.bb = BaseChessBoard(FULL_INIT_FEN)
        self.red = True
    
    def get_side(self):
        return "red" if self.red else "black"
    
    def make_move(self,move):
        i = move
        x1,y1,x2,y2 = int(i[0]),int(i[1]),int(i[3]),int(i[4])
        #boardarr = bb.get_board_arr()
        if self.red:
            moveresult = self.bb.move(Pos(x1,y1),Pos(x2,y2))
        else:
            moveresult = self.bb.move(Pos(x1,9-y1),Pos(x2,9-y2))
        assert(moveresult != None)
        self.red = not self.red
    
    def print_board(self):
        self.bb.print_board()
    
    def get_board_arr(self):
        feature_list = {"red":['A', 'B', 'C', 'K', 'N', 'P', 'R']
                             ,"black":['a', 'b', 'c', 'k', 'n', 'p', 'r']}
        # chess picker features
        picker_x = []
        picker_y = []
        boardarr = self.bb.get_board_arr()
        if self.red:
            for one in feature_list['red']:
                picker_x.append(np.asarray(boardarr == one,dtype=np.uint8))
            for one in feature_list['black']:
                picker_x.append(np.asarray(boardarr == one,dtype=np.uint8))
        else:
            for one in feature_list['black']:
                picker_x.append(np.asarray(boardarr == one,dtype=np.uint8))
            for one in feature_list['red']:
                picker_x.append(np.asarray(boardarr == one,dtype=np.uint8))
        picker_x = np.asarray(picker_x)
        if self.red:
            return picker_x
        else:
            return picker_x[:,::-1,:]