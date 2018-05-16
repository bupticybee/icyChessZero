# -*- coding: utf-8 -*-

'''
Copyright (C) 2014  walker li <walker8088@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import sys
import copy
import numpy as np

from cchess.exception import *
from cchess.piece import *
from cchess.move import *

#-----------------------------------------------------#
FULL_INIT_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'

#-----------------------------------------------------#
_text_board = [
#u' 1  2   3   4   5   6   7   8   9',
u'0 ┌─┬─┬─┬───┬─┬─┬─┐',
u'  │  │  │  │＼│／│　│　│　│',
u'1 ├─┼─┼─┼─※─┼─┼─┼─┤',
u'  │　│　│　│／│＼│　│　│　│',
u'2 ├─┼─┼─┼─┼─┼─┼─┼─┤',
u'  │　│　│　│　│　│　│　│　│',
u'3 ├─┼─┼─┼─┼─┼─┼─┼─┤',
u'  │　│　│　│　│　│　│　│　│',
u'4 ├─┴─┴─┴─┴─┴─┴─┴─┤',
u'  │　                         　 │',
u'5 ├─┬─┬─┬─┬─┬─┬─┬─┤',
u'  │　│　│　│　│　│　│　│　│',
u'6 ├─┼─┼─┼─┼─┼─┼─┼─┤',
u'  │　│　│　│　│　│　│　│　│',
u'7 ├─┼─┼─┼─┼─┼─┼─┼─┤',
u'  │　│　│　│＼│／│　│　│　│',
u'8 ├─┼─┼─┼─※─┼─┼─┼─┤',
u'  │　│　│　│／│＼│　│　│　│',
u'9 └─┴─┴─┴───┴─┴─┴─┘',
u'  0   1   2   3   4   5   6   7   8'
#u'  九 八  七  六  五  四  三  二  一'
]

_fench_txt_name_dict = {
   'K': u"帅",
   'k': u"将",
   'A': u"仕",
   'a': u"士",
   'B': u"相", 
   'b': u"象",
   'N': u"马",
   'n': u"碼",
   'R': u"车",
   'r': u"砗",
   'C': u"炮", 
   'c': u"砲",
   'P': u"兵", 
   'p': u"卒"     

   }
#-----------------------------------------------------#

def _pos_to_text_board_pos(pos):
    return Pos(2*pos.x+2, (9 - pos.y)*2)     

def _fench_to_txt_name(fench) :
    return _fench_txt_name_dict[fench]
     
#-----------------------------------------------------#
class BaseChessBoard(object) :
    def __init__(self, fen = None):
        self.clear()
        if fen: self.from_fen(fen)
                
    def clear(self):    
        self._board = [[None for x in range(9)] for y in range(10)]
        self.move_side = ChessSide.RED 
        
    def copy(self):
        return copy.deepcopy(self)
        
    def put_fench(self, fench, pos):
        if self._board[pos.y][pos.x] != None:
            return False    
        
        self._board[pos.y][pos.x] = fench
        
        return True
        
    def get_fench(self, pos):
        return self._board[pos.y][pos.x]
    
    def get_piece(self, pos): 
        fench = self._board[pos.y][pos.x]
        
        if not fench:
                return None
        
        return Piece.create(self, fench, pos)
        
    def is_valid_move_t(self, move_t):
        pos_from, pos_to = move_t
        return self.is_valid_move(pos_from, pos_to)
        
    def is_valid_move(self, pos_from, pos_to):
        
        '''
        只进行最基本的走子规则检查，不对每个子的规则进行检查，以加快文件加载之类的速度
        '''
        
        if not (0 <= pos_to.x <= 8): return False
        if not (0 <= pos_to.y <= 9): return False
        
        fench_from = self._board[pos_from.y][pos_from.x]
        if not fench_from :
            return False        
            
        _, from_side = fench_to_species(fench_from)
        
        #move_side 不是None值才会进行走子颜色检查，这样处理某些特殊的存储格式时会处理比较迅速
        if self.move_side and (from_side != self.move_side) :
            return False        
            
        fench_to = self._board[pos_to.y][pos_to.x]
        if not fench_to :
            return True 
            
        _, to_side = fench_to_species(fench_to)
        
        return (from_side != to_side) 
    
    def _move_piece(self, pos_from, pos_to):
        
        fench = self._board[pos_from.y][pos_from.x]
        self._board[pos_to.y][pos_to.x] = fench
        self._board[pos_from.y][pos_from.x] = None
        
        return fench
        
    def move(self, pos_from, pos_to):
        pos_from.y = 9 - pos_from.y
        pos_to.y = 9 - pos_to.y
        if not self.is_valid_move(pos_from, pos_to):
             return None 
             
        board = self.copy()
        fench = self.get_fench(pos_to)
        self._move_piece(pos_from, pos_to)
                
        return Move(board, pos_from, pos_to)
        
    def move_iccs(self,move_str):
        move_from, move_to = Move.from_iccs(move_str)
        return move(move_from, move_to)
    
    def move_chinese(self,move_str):
        move_from, move_to = Move.from_chinese(self, move_str)
        return move(move_from, move_to)
    
    def next_turn(self) :
        if self.move_side == None :
            return None
            
        self.move_side = ChessSide.next_side(self.move_side)
       
        return self.move_side
        
    def from_fen(self, fen):
        
        num_set = set(('1', '2', '3', '4', '5', '6', '7', '8', '9'))
        ch_set = set(('k','a','b','n','r','c','p'))
        
        self.clear()
        
        if not fen or fen == '':
                return
        
        fen = fen.strip()
        
        x = 0
        y = 9

        for i in range(0, len(fen)):
            ch = fen[i]
            
            if ch == ' ': break
            elif ch == '/':
                y -= 1
                x = 0
                if y < 0: break
            elif ch in num_set:
                x += int(ch)
                if x > 8: x = 8
            elif ch.lower() in ch_set:
                if x <= 8:
                    self.put_fench(ch, Pos(x, y))                
                    x += 1
            else:
                return False
        
        fens = fen.split() 
        
        self.move_side = None
        if (len(fens) >= 2) and (fens[1] == 'b') :
                 self.move_side = ChessSide.BLACK
        else:
                self.move_side = ChessSide.RED            
        
        if len(fens) >= 6  :
                self.round = int(fens[5])
        else:
                self.round = 1      
                
        return True 
    
    def count_x_line_in(self, y, x_from, x_to):
        return reduce(lambda count, fench: count+1 if fench else count, self.x_line_in(y, x_from, x_to), 0)
    
    def count_y_line_in(self, x, y_from, y_to):
        return reduce(lambda count, fench: count+1 if fench else count, self.y_line_in(x, y_from, y_to), 0)
        
    def x_line_in(self, y, x_from, x_to):
        step = 1 if x_to > x_from else -1
        return [ self._board[y][x] for x in range(x_from+step, x_to, step) ]
    
    def y_line_in(self, x, y_from, y_to):
        step = 1 if y_to > y_from else -1
        return [ self._board[y][x] for y in range(y_from+step, y_to, step) ]
    
    def to_fen(self):
        return self.to_short_fen() + ' - - 0 1'
        
    def to_short_fen(self):
        fen = ''
        count = 0
        for y in range(9, -1, -1):
            for x in range(9):
                fench = self._board[y][x]
                if fench:        
                    if count is not 0:
                        fen += str(count)
                        count = 0
                    fen += fench
                else:
                    count += 1
                    
            if count > 0:
                fen += str(count)
                count = 0
                
            if y > 0: fen += '/'
                        
        if self.move_side is ChessSide.BLACK:
            fen += ' b'
        elif self.move_side is ChessSide.RED :
            fen += ' w'
        else :
            raise CChessException('Move Side Error' + str(self.move_side))
        
        return fen

    def dump_board(self):
                
        board_str = _text_board[:]
    
        y = 0
        for line in self._board:
            x = 0
            for ch in line:
                if ch : 
                        pos = _pos_to_text_board_pos(Pos(x,y))
                        new_text=board_str[pos.y][:pos.x] + _fench_to_txt_name(ch) + board_str[pos.y][pos.x+1:]
                        board_str[pos.y] = new_text
                x += 1                         
            y += 1
        
        return board_str
        
    def print_board(self):
    
        board_txt = self.dump_board()
        print()
        for line in board_txt:
                print(line)
        print()
        
    def get_board_arr(self):
        return np.asarray(self._board[::-1])
        
#-----------------------------------------------------#

class ChessBoard(BaseChessBoard):
    def __init__(self, fen = None):        
        super(ChessBoard, self).__init__(fen)
    
    def put_fench(self, fench, pos):
        self._board[pos.y][pos.x] = fench
        
    def is_valid_move(self, pos_from, pos_to):
        if not super(ChessBoard, self).is_valid_move(pos_from, pos_to):
                return False
        
        piece = self.get_piece(pos_from)
        if not piece.is_valid_move(pos_to):
            return False
        return True
         
    def is_checked_move(self, pos_from, pos_to):
        board = self.copy()
        board._move_piece(pos_from, pos_to)
        return board.is_checked()
        
    def is_checked(self):
        king = self.get_king(self.move_side)
        king.create_moves()
        if not king : return 0
        killers = self.get_side_pieces(ChessSide.next_side(self.move_side))
        '''
        for piece in killers:
                if piece.is_valid_move(Pos(king.x, king.y)):
                        #print piece.x, piece.y, piece.fench
                        return 1
        return 0
        '''        
        return reduce(lambda count, piece : count+1 if piece.is_valid_move(Pos(king.x, king.y)) else count, killers, 0)                
        
    def is_checkmate(self):
        defenders = self.get_side_pieces(self.move_side)
        for piece in defenders :
            for move_it in piece.create_moves():
                if self.is_valid_move_t(move_it):
                        if not self.is_checked_move(move_it[0], move_it[1]):
                                return False
        return True
        
    def get_king(self, side):
        limit_y = ((0,1,2), (7,8,9))
        for x in (3,4,5):
                for y in limit_y[side]:
                        fench = self._board[y][x]
                        if not fench :
                                continue
                        if fench.lower() == 'k':
                                return  Piece.create(self, fench, Pos(x,y))
        return None
    
    def get_side_pieces(self, side):
        pieces = [] 
        for x in range(9):
                for y in range(10):
                        fench = self._board[y][x]
                        if not fench :
                                continue
                        _, p_side = fench_to_species(fench)
                        if p_side == side :        
                                pieces.append(Piece.create(self, fench, Pos(x,y)))
        return pieces
        
        
#-----------------------------------------------------#
if __name__ == '__main__':
    pass
#        
#        board = ChessBoard(FULL_INIT_FEN)
#        board.print_board()
#        
#        k = board.get_king(ChessSide.RED)
#        print (k.x, k.y) == (4,0)
#        k = board.get_king(ChessSide.BLACK)
#        print (k.x, k.y) == (4,9)
#        
#        print board.x_line_in(0, 0, 8)
#        print board.x_line_in(0, 8, 0)
#        #print board.x_line_in(9, 0, 10)
#        print board.y_line_in(4, -1, 10)
#        print board.y_line_in(4, 10, -1)
#        
#        print board.count_x_line_in(0, 0, 8) == 7
#        print board.count_y_line_in(4,0,9) == 2
#        print board.count_y_line_in(4,1,8) == 2
#        
#        print board.is_checked()
#        
#        print board.copy().move(Pos(7,2),Pos(4,2)).to_chinese() == u'炮二平五'
#        print board.copy().move(Pos(1,2),Pos(1,1)).to_chinese() == u'炮八退一'
#        print board.copy().move(Pos(7,2),Pos(7,6)).to_chinese() == u'炮二进四'
#        print board.copy().move(Pos(7,7),Pos(4,7)).to_chinese() == u'炮８平５'
#        print board.copy().move(Pos(7,7),Pos(7,3)).to_chinese() == u'炮８进４'
#        print board.copy().move(Pos(6,3),Pos(6,4)).to_chinese() == u'兵三进一'
#        print board.copy().move(Pos(8,0),Pos(8,1)).to_chinese() == u'车一进一'
#        print board.copy().move(Pos(0,9),Pos(0,8)).to_chinese() == u'车１进１'
#        print board.copy().move(Pos(4,0),Pos(4,1)).to_chinese() == u'帅五进一'
#        print board.copy().move(Pos(4,9),Pos(4,8)).to_chinese() == u'将５进１'
#        print board.copy().move(Pos(2,0),Pos(4,2)).to_chinese() == u'相七进五'
#        print board.copy().move(Pos(5,0),Pos(4,1)).to_chinese() == u'仕四进五'
#        print board.copy().move(Pos(7,0),Pos(6,2)).to_chinese() == u'马二进三'
#                 