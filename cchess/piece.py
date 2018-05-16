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

#from sets import *
from enum import *
    
#-----------------------------------------------------#

h_level_index = \
(
        (u"九",u"八",u"七",u"六",u"五",u"四",u"三",u"二",u"一"), 
        (u"１",u"２",u"３",u"４",u"５",u"６",u"７",u"８",u"９") 
)

v_change_index = \
(
        (u"错", ""u"一", u"二", u"三", u"四", u"五", u"六", u"七", u"八", u"九"), 
        (u"误", ""u"１", u"２", u"３", u"４", u"５", u"６", u"７", u"８", u"９")
)

#-----------------------------------------------------#

advisor_pos = (
    ((3, 0), (5, 0), (4, 1), (3, 2), (5, 2)),
    ((3, 9), (5, 9), (4, 8), (3, 7), (5, 7)), 
    )

bishop_pos = (
    ((2, 0), (6, 0), (0, 2), (4, 2), (9, 2), (2, 4), (6, 4)),
    ((2, 9), (6, 9), (0, 7), (4, 7), (9, 7), (2, 5), (6, 5)), 
    )

#-----------------------------------------------------#
class ChessSide(IntEnum):     
    RED = 0
    BLACK = 1
    
    @staticmethod
    def next_side(side):
        return {ChessSide.RED:ChessSide.BLACK, ChessSide.BLACK:ChessSide.RED}[side]
     
#-----------------------------------------------------#
class PieceT(IntEnum):
    KING = 1
    ADVISOR = 2
    BISHOP = 3
    KNIGHT = 4
    ROOK = 5 
    CANNON = 6
    PAWN = 7
        
#-----------------------------------------------------#
fench_species_dict = {
    'k': PieceT.KING,
    'a': PieceT.ADVISOR,
    'b': PieceT.BISHOP,
    'n': PieceT.KNIGHT,
    'r': PieceT.ROOK,
    'c': PieceT.CANNON,
    'p': PieceT.PAWN
}

fench_name_dict = {
   'K': u"帅",
   'k': u"将",
   'A': u"仕",
   'a': u"士",
   'B': u"相", 
   'b': u"象",
   'N': u"马",
   'n': u"马",
   'R': u"车",
   'r': u"车",
   'C': u"炮", 
   'c': u"炮",
   'P': u"兵", 
   'p': u"卒"     
}

    
species_fench_dict = {
    PieceT.KING:    ('K', 'k'),
    PieceT.ADVISOR: ('A', 'a'),
    PieceT.BISHOP:  ('B', 'b'),
    PieceT.KNIGHT:  ('N', 'n'),
    PieceT.ROOK:    ('R', 'r'),
    PieceT.CANNON:  ('C', 'c'),
    PieceT.PAWN:    ('P', 'p')     
}

#-----------------------------------------------------#
def fench_to_chinese(fench) :
    return fench_name_dict[fench]
    
def fench_to_species(fen_ch):
    return fench_species_dict[fen_ch.lower()], ChessSide.BLACK if fen_ch.islower() else ChessSide.RED
    
def species_to_fench(species, side):
    return species_fench_dict[species][side]
    
#KING, ADVISOR, BISHOP, KNIGHT, ROOK, CANNON, PAWN

chessman_show_name_dict = {
    PieceT.KING:    (u"帅", u"将"),
    PieceT.ADVISOR: (u"仕", u"士"),
    PieceT.BISHOP:  (u"相", u"象"),
    PieceT.KNIGHT:  (u"马", u"碼"),
    PieceT.ROOK:    (u"车", u"砗"),
    PieceT.CANNON:  (u"炮", u"砲"),
    PieceT.PAWN:    (u"兵", u"卒")     
}

def get_show_name(species, side) :
        return chessman_show_name_dict[species][side]
        
    
#-----------------------------------------------------#
class Pos(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def abs_diff(self, other):
        return (abs(self.x - other.x), abs(self.y - other.y)) 
    
    def middle(self, other):
        return Pos((self.x + other.x) / 2, (self.y + other.y) / 2)
        
    def __str__(self):
        return str(self.x) + ":" + str(self.y)
    
    def __eq__(self, other): 
        return (self.x == other.x) and (self.y == other.y)
    
    def __ne__(self, other): 
        return (self.x != other.x) or (self.y != other.y)
    
    def __call__(self):
        return (self.x, self.y)
        
#-----------------------------------------------------#
        
class Piece(object):
   
    def __init__(self, board, fench, pos):
        
        self.board = board
        self.fench = fench
        
        species, side = fench_to_species(fench)  
        
        self.species = species
        self.side = side
        
        self.x, self.y = pos()
        
    def is_valid_pos(self, pos):
        return True
        
    def is_valid_move(self, pos):
        return True 
    
    @staticmethod
    def create(board, fench, pos):    
        p_type = fench.lower()
        if p_type == 'k':
                return King(board, fench, pos)
        if p_type == 'a':
                return Advisor(board, fench, pos)
        if p_type == 'b':
                return Bishop(board, fench, pos)
        if p_type == 'r':
                return Rook(board, fench, pos)
        if p_type == 'c':
                return Cannon(board, fench, pos)
        if p_type == 'n':
                return Knight(board, fench, pos)
        if p_type == 'p':
                return Pawn(board, fench, pos)
                
    
    '''    
    def chinese_move_to_std_move(self, move_str):
        
        if self.species in self.__chinese_move_to_std_move_checks :
            new_pos = self.__chinese_move_to_std_move_checks[self.species](move_str)
        else :    
            new_pos = self.__chinese_move_to_std_move_default(move_str)
        
        if not new_pos :
            return None
            
        if not self.can_move_to(new_pos[0] , new_pos[1]):
            return None 
            
        return ((self.x,  self.y), new_pos)
    
    
    def __chinese_move_to_std_move_advisor(self, move_str):
    
        if move_str[0] == u"平":
            return None
        
        new_x = h_level_index[self.side].index(move_str[1])
                
        if move_str[0] == u"进" :
            diff_y = -1
        elif move_str[0] == u"退" :
            diff_y = 1    
        else :
            return None
        
        if self.side == ChessSide.BLACK:
            diff_y = - diff_y
        
        new_y = self.y - diff_y
        
        return (new_x,  new_y)
        
    def __chinese_move_to_std_move_bishop(self, move_str):
        if move_str[0] == u"平":
            return None
        
        new_x = h_level_index[self.side].index(move_str[1])
                
        if move_str[0] == u"进" :
            diff_y = -2
        elif move_str[0] == u"退" :
            diff_y = 2    
        else :
            return None
        
        if self.side == ChessSide.BLACK:
            diff_y = - diff_y
        
        new_y = self.y - diff_y
        
        return (new_x,  new_y)
        
    def __chinese_move_to_std_move_knight(self, move_str):
        if move_str[0] == u"平":
            return None
        
        new_x = h_level_index[self.side].index(move_str[1])
        
        diff_x = abs(self.x - new_x)
        
        if move_str[0] == u"进" :
            diff_y = [3, 2, 1][diff_x]
            
        elif move_str[0] == u"退" :
            diff_y = [-3, -2, -1][diff_x]
            
        else :
            return None
        
        if self.side == ChessSide.RED:
            diff_y = -diff_y
        
        new_y = self.y - diff_y
        
        return (new_x,  new_y)
        
    def __chinese_move_to_std_move_default(self, move_str):
        
        if move_str[0] == u"平":
            new_x = h_level_index[self.side].index(move_str[1])
            
            return (new_x,  self.y)
            
        else :
            #王，车，炮，兵的前进和后退
            diff = v_change_index[self.side].index(move_str[1])
            
            if move_str[0] == u"退":
                diff = -diff
            elif move_str[0] != u"进":
                return None
                
            if self.side == ChessSide.BLACK:
                diff = -diff
            
            new_y = self.y + diff
            
            return (self.x,  new_y)
    ''' 
#-----------------------------------------------------#        
#王
class King(Piece):
        
    def is_valid_pos(self, pos):
        
        if pos.x < 3 or pos.x > 5:
            return False
            
        if (self.side == ChessSide.RED) and pos.y > 2:
            return False
            
        if (self.side == ChessSide.BLACK) and pos.y < 7:
            return False
            
        return True
        
    def is_valid_move(self, pos):
        
        #先检查王吃王
        k2 = self.board.get_king(ChessSide.next_side(self.side))
        
        if ((k2.x,k2.y) == pos()) and self.x == k2.x:
            count = self.board.count_y_line_in(self.x, self.y, k2.y)    
            if count == 0:
                return True
                
        if not self.is_valid_pos(pos) :
            return False
                
        diff = pos.abs_diff(Pos(self.x, self.y))
        
        return True if ((diff[0] + diff[1]) == 1) else False
    
    def create_moves(self):
        poss = [Pos(self.x+1,self.y),Pos(self.x-1,self.y),Pos(self.x,self.y+1),Pos(self.x,self.y-1)]
        curr_pos = Pos(self.x, self.y)
        moves = [(curr_pos, to_pos) for to_pos in poss]
        return filter(self.board.is_valid_move_t, moves)
                
#-----------------------------------------------------#    
#士
class Advisor(Piece): 
   
    def is_valid_pos(self, pos):
         return True if pos() in advisor_pos[self.side] else False
       
    def is_valid_move(self, pos):
        
        if not self.is_valid_pos(pos) :
            return False
        
        if Pos(self.x, self.y).abs_diff(pos) == (1,1):
            return True
        
        return False
    
    def create_moves(self):
        poss = [Pos(self.x+1,self.y+1),Pos(self.x+1,self.y-1),Pos(self.x-1,self.y+1),Pos(self.x-1,self.y-1)]
        curr_pos = Pos(self.x, self.y)
        moves = [(curr_pos, to_pos) for to_pos in poss]
        return filter(self.board.is_valid_move_t, moves)
            
#-----------------------------------------------------#    
#象    
class Bishop(Piece): 
    def is_valid_pos(self, pos):
        return True if pos() in bishop_pos[self.side] else False
    
    def is_valid_move(self, pos):
        
        if Pos(self.x, self.y).abs_diff(pos) != (2,2):
            return False
          
        #塞象眼检查
        if self.board.get_fench(Pos(self.x, self.y).middle(pos)) != None :
            return False
        
        return True
    
    def create_moves(self):
        poss = [Pos(self.x+2,self.y+2),Pos(self.x+2,self.y-2),Pos(self.x-2,self.y+2),Pos(self.x-2,self.y-2)]
        curr_pos = Pos(self.x, self.y)
        moves = [(curr_pos, to_pos) for to_pos in poss]
        return filter(self.board.is_valid_move_t, moves)
        
#-----------------------------------------------------#    
#马
class Knight(Piece): 
    def is_valid_move(self, pos):
        
        if (abs(self.x - pos.x) == 2) and (abs(self.y - pos.y) == 1):
            
            m_x = (self.x + pos.x) / 2
            m_y = self.y
            
            #别马腿检查
            if self.board.get_fench(Pos(m_x, m_y)) == None :
                return True

        if (abs(self.x - pos.x) == 1) and (abs(self.y - pos.y) == 2):
            
            m_x = self.x
            m_y = (self.y + pos.y) / 2
            
            #别马腿检查
            if self.board.get_fench(Pos(m_x, m_y)) == None :
                return True

        return False
    
    def create_moves(self):
        poss = [Pos(self.x+1,self.y+2),Pos(self.x+1,self.y-2),
                Pos(self.x-1,self.y+2),Pos(self.x-1,self.y-2),
                Pos(self.x+2,self.y+1),Pos(self.x+2,self.y-1),
                Pos(self.x-2,self.y+1),Pos(self.x-2,self.y-1),
                ]
        curr_pos = Pos(self.x, self.y)
        moves = [(curr_pos, to_pos) for to_pos in poss]
        return filter(self.board.is_valid_move_t, moves)
        
#-----------------------------------------------------#    
#车
class Rook(Piece): 
    def is_valid_move(self, pos):    
        if self.x != pos.x:
            #斜向移动是非法的
            if self.y != pos.y:   
                return False
                
            #水平移动
            if self.board.count_x_line_in(self.y, self.x, pos.x) == 0:
                return True
                
        else :
            #垂直移动
            if self.board.count_y_line_in(self.x, self.y, pos.y) == 0:
                return True
                
        return False
        
    def create_moves(self):
        moves = []
        curr_pos = Pos(self.x, self.y)
        for x in range(9):
                for y in range(10):
                        if self.x == x and self.y == y:
                                continue
                        moves.append((curr_pos, Pos(x,y)))                                
        return filter(self.board.is_valid_move_t, moves)
        
#-----------------------------------------------------#    
#炮
class Cannon(Piece): 
    def is_valid_move(self, pos):
        
        if self.x != pos.x:
            #斜向移动是非法的
            if self.y != pos.y:   
                return False
            
            #水平移动    
            count = self.board.count_x_line_in(self.y, self.x, pos.x)
            if (count == 0) and (self.board.get_fench(pos) == None):
                return True
            if (count == 1) and (self.board.get_fench(pos) != None):
                return True
        else :
            #垂直移动
            count = self.board.count_y_line_in(self.x, self.y, pos.y)
            if (count == 0) and (self.board.get_fench(pos) == None):
                return True
            if (count == 1) and (self.board.get_fench(pos) != None):
                return True
             
        return False
        
    def create_moves(self):
        moves = []
        curr_pos = Pos(self.x, self.y)
        for x in range(9):
                for y in range(10):
                        if self.x == x and self.y == y:
                                continue
                        moves.append((curr_pos, Pos(x,y)))                                
        return filter(self.board.is_valid_move_t, moves)
        
#-----------------------------------------------------#    
#兵/卒
class Pawn(Piece): 
    def is_valid_pos(self, pos):
        
        if (self.side == ChessSide.RED) and pos.y < 3:
            return False
            
        if (self.side == ChessSide.BLACK) and pos.y > 6:
            return False
            
        return True
    
    def is_valid_move(self, pos):
        
        not_over_river_step = ((0, 1), (0, -1))
        over_river_step = (((-1, 0), (1, 0), (0, 1)),((-1, 0), (1, 0), (0, -1)))
                           
        step = (pos.x - self.x, pos.y - self.y)
        
        over_river = self.is_over_river()
        
        if (not over_river) and (step == not_over_river_step[self.side]):
                return True
        
        if over_river and (step in over_river_step[self.side]):
                return True
                
        return False
    
    def is_over_river(self) :      
        if (self.side == ChessSide.RED) and (self.y > 4) :
            return True
            
        if (self.side == ChessSide.BLACK) and (self.y < 5) :
            return True
            
        return False
        
    def create_moves(self):
        moves = []
        curr_pos = Pos(self.x, self.y)
        for x in range(9):
                for y in range(10):
                        if self.x == x and self.y == y:
                                continue
                        moves.append((curr_pos, Pos(x,y)))                                
        return filter(self.board.is_valid_move_t, moves)
                               
#-----------------------------------------------------#    
if __name__ == '__main__':
    pass 
    
    