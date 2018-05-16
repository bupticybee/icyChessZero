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

import os

from xml.etree import ElementTree as et

from cchess.board import *
from cchess.game import *
from cchess.exception import *

#-----------------------------------------------------#
                
def read_from_cbf(file_name):
        
        def decode_move(move_str):
            p_from = Pos(int(move_str[0]), 9 - int(move_str[1])) 
            p_to = Pos(int(move_str[3]),   9 - int(move_str[4])) 
            
            return (p_from, p_to)
            
        tree = et.parse(file_name)
        root = tree.getroot()
        
        head = root.find("Head")
        for node in head.getchildren() :
            if node.tag == "FEN":
                init_fen = node.text
            #print node.tag
        
        books = {}
        board = BaseChessBoard(init_fen)
        
        move_list = root.find("MoveList").getchildren()
        
        game = Game(board)
        last_move = game
        step_no = 1    
        for node in move_list[1:] : 
                move_from, move_to = decode_move(node.attrib["value"])           
                if board.is_valid_move(move_from, move_to) :
                        new_move = board.move(move_from, move_to)    
                        last_move.append_next_move(new_move)
                        last_move = new_move
                        board.next_turn()
                else :
                        raise CChessException("bad move at %d %s %s" % (step_no, move_from, move_to))
                step_no += 1        
        return game
                
#-----------------------------------------------------#

if __name__ == '__main__':
    game = read_from_cbf('test\\test.cbf')
    game.print_init_board()
    game.print_chinese_moves()
    
