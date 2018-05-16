import xmltodict
from cchess import *
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