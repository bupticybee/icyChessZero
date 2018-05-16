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
import struct


from bs4 import BeautifulSoup
import xml.etree.cElementTree as et

from cchess.board import *
from cchess.game import *
from cchess.exception import *

#-----------------------------------------------------#
def read_from_dhtml(html_page):
        res_dict = __parse_dhtml(html_page)
        game = read_from_txt(res_dict['moves'], res_dict['init']) 
        return game
        
#-----------------------------------------------------#                   
def read_from_txt(moves_txt, pos_txt = None):
        
        def decode_txt_pos(pos) :
                return Pos(int(pos[0]), 9-int(pos[1])) 

        #车马相士帅士相马车炮炮兵兵兵兵兵
        #车马象士将士象马车炮炮卒卒卒卒卒
        chessman_kinds = 'RNBAKABNRCCPPPPP'  
        
        if not pos_txt:
            board = BaseChessBoard(FULL_INIT_FEN)
        else:
            if len(pos_txt) != 64:
                 raise CChessException("bad pos_txt")
                 
            board = BaseChessBoard()
            for side in range(2):
                for man_index in range(16):
                        pos_index = (side * 16 + man_index)*2 
                        man_pos = pos_txt[pos_index : pos_index + 2]
                        if man_pos == '99':
                                continue
                        pos = decode_txt_pos(man_pos)  
                        fen_ch = chr(ord(chessman_kinds[man_index]) + side * 32)
                        board.put_fench(fen_ch, pos)
        
        last_move = None
        if not moves_txt:
            return Game(board)
        step_no = 0
        while step_no*4 < len(moves_txt) : 
                steps = moves_txt[step_no*4:step_no*4+4]
                
                move_from = decode_txt_pos(moves_txt[step_no*4:step_no*4+2])
                move_to = decode_txt_pos(moves_txt[step_no*4+2:step_no*4+4])
                
                if board.is_valid_move(move_from, move_to) :
                        
                        if not last_move:
                            _, man_side = fench_to_species(board.get_fench(move_from))
                            board.move_side = man_side
                            game = Game(board)
                            last_move = game
                            
                        new_move = board.move(move_from, move_to)    
                        last_move.append_next_move(new_move)
                        last_move = new_move
                        board.next_turn()
                else :
                        raise CChessException("bad move at %d %s %s" % (step_no, move_from, move_to))
                step_no += 1
        if step_no == 0:
                game = Game(board)
                
        return game
                
#-----------------------------------------------------#
def __str_between(src, begin_str, end_str) :
        first = src.find(begin_str) + len(begin_str)
        last = src.find(end_str)
        if (first != -1) and (last != -1) :
                return src[first:last]
        else :
                return None

def __str_between2(src, begin_str, end_str) :
        first = src.find(begin_str) + len(begin_str)
        last = src.find(end_str)
        if last > first:
                return src[first:last]
        if last == -1:
                return None
                
        src2 = src[last + len(end_str):]        
        f2 = src2.find(begin_str) + len(begin_str)
        l2 = src2.find(end_str)
        if l2 > f2:
                return src2[f2:l2]
        else :
                return None
                
def __parse_dhtml(html_page) :        
        result_dict = {}
        text = html_page.decode('GB18030')
        result_dict['event'] = __str_between(text, '[DhtmlXQ_event]', '[/DhtmlXQ_event]')
        if result_dict['event'] :
                result_dict['event'] = result_dict['event']
        
        result_dict['title'] = __str_between(text, '[DhtmlXQ_title]', '[/DhtmlXQ_title]')
        if result_dict['title'] :
                result_dict['title'] = result_dict['title']
        
        result_dict['result'] = __str_between(text, '[DhtmlXQ_result]', '[/DhtmlXQ_result]')
        if result_dict['result'] :
                result_dict['result'] = result_dict['result']
           
        init = __str_between(text, '[DhtmlXQ_binit]', '[/DhtmlXQ_binit]')
        result_dict['init'] = init.encode('utf-8') if init else None
        moves = __str_between2(text, '[DhtmlXQ_movelist]', '[/DhtmlXQ_movelist]')
        result_dict['moves'] = moves.encode('utf-8') if moves else None
                
        return result_dict
        
#-----------------------------------------------------#
if __name__ == '__main__':

        pos_s = '9999999949399920109981999999993129629999409999997109993847999999'
#
#        move_s = '31414050414050402032' #'77477242796770628979808166658131192710222625120209193136797136267121624117132324191724256755251547431516212226225534222434532454171600105361545113635161636061601610'
#      
#        try:
#            game = read_from_txt(moves_txt = move_s, pos_txt = pos_s)
#        except CChessException as e:
#            print e.reason
#        else:    
#        
#            board_txt = game.dump_init_board()
#            print game.init_board.to_fen()
#            print
#            for line in board_txt:
#                print line
#            print   
#        
#            moves = game.dump_std_moves() 
#            print moves             
#            for it in moves[0]:
#                print it
#        