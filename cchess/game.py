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

from cchess.board import *
from cchess.move  import *
                
    
# 比赛结果
UNKNOWN, RED_WIN, BLACK_WIN, PEACE  =  range(4)
result_str = (u"未知", u"红胜", u"黑胜",  u"平局" ) 

#存储类型
BOOK_UNKNOWN, BOOK_ALL, BOOK_BEGIN, BOOK_MIDDLE, BOOK_END = range(5)
book_type_str = (u"未知", u"全局", u"开局",  u"中局",  u"残局") 
    
                
#-----------------------------------------------------#
class Game(object):
    def __init__(self, board = None, annotation = None):
        self.init_board = board.copy()
        self.annotation = annotation
        self.next_move = None
        self.infos = {}
        
    def append_next_move(self, chess_move):
        chess_move.parent = self
        if not self.next_move:
                self.next_move = chess_move
        else:
                #找最右一个        
                move = self.next_move
                while move.right_move:
                        move = move.right_move
                move.right_move = chess_move 
    
    def verify_moves(self):
        return True 
        move_list = self.dump_moves() 
        for move_line in move_list:        
            j = 0
            for move in move_line:   
                if not move.is_valid_move():
                    #print moves_to_chinese(self.init_fen, move_line[:j])
                    #print j, move, move_line
                    return False
                j += 1
        return True
    
    def dump_init_board(self):
        return self.init_board.dump_board()
        
    def dump_moves(self):
        
        if not self.next_move:
                return []

        move_list = []
        curr_move = []
        move_list.append(curr_move)

        self.next_move.dump_moves(move_list, curr_move)
        
        return move_list
    
    def dump_std_moves(self):
        return [[str(move) for move in move_line] for move_line in self.dump_moves()]
        
    def dump_chinese_moves(self):
        return [[move.to_chinese() for move in move_line] for move_line in self.dump_moves()]
      
    def print_init_board(self):
        self.init_board.print_board()
            
    def print_chinese_moves(self, steps_per_line=3):
    
        moves = self.dump_chinese_moves()
        line_no = 1
        for line in moves:
                
                if len(moves) > 1:
                    print (u'第%d分支' % line_no)
                    
                i = 0
                for it in line:
                        if (i%2) == 0:
                                print ('%2d. '%(i/2+1),endl='')
                        print (it,endl='') 
                        i += 1
                        if (i%(steps_per_line*2)) == 0:
                                print()
                print
                line_no += 1
                
    def dump_info(self):   
        for key in self.info:
                print (key, self.info[key])
        
#-----------------------------------------------------#
if __name__ == '__main__': 
    from reader_xqf import *   
    game = read_from_xqf('test\\ucci_test1.xqf')
    game.init_board.move_side = ChessSide.RED
    game.print_init_board()
    game.print_chinese_moves()
