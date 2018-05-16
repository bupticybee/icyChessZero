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

import sys,time
from enum import *

from subprocess import PIPE, Popen
from threading import Thread

#from Queue import Queue, Empty
#from multiprocessing import Queue,Empty
from cchess.board import *
from cchess.move import *

#-----------------------------------------------------#

#Engine status   
class EngineStatus(IntEnum):
        BOOTING = 1,
        READY = 2, 
        WAITING = 3, 
        INFO_MOVE = 4, 
        MOVE = 5, 
        DEAD = 6, 
        UNKNOWN = 7, 
        BOARD_RESET = 8
    
ON_POSIX = 'posix' in sys.builtin_module_names

#-----------------------------------------------------#

class UcciEngine(Thread):
    def __init__(self, name = ''):
        super(UcciEngine, self).__init__()
        
        self.engine_name = name
        
        self.daemon  = True
        self.running = False
        
        self.engine_status = None
        self.ids = []
        self.options = []
        
        self.last_fen = None
        self.move_queue = Queue()
    
    def run(self) :
        
        self.running = True
        
        while self.running :
            output = self.pout.readline().strip()
            self.engine_out_queque.put(output)
            
    def handle_msg_once(self) :
        try:  
            output = self.engine_out_queque.get_nowait()
        except Empty:
            return False
    
        if output in ['bye','']: #stop pipe
            self.pipe.terminate()
            return False
            
        self.__handle_engine_out_line(output)
        
        return True
    
    def load(self, engine_path):
    
        self.engine_name = engine_path
        
        try:
            self.pipe = Popen(self.engine_name, stdin=PIPE, stdout=PIPE)#, close_fds=ON_POSIX)
        except OSError:
            return False
            
        time.sleep(0.5)
        
        (self.pin, self.pout) = (self.pipe.stdin,self.pipe.stdout)
        
        self.engine_out_queque = Queue()
        
        self.enging_status = EngineStatus.BOOTING
        self.send_cmd("ucci")
        
        self.start()
        
        while self.enging_status ==  EngineStatus.BOOTING :
            self.handle_msg_once()
            
        return True
        
    def quit(self):
        
        self.send_cmd("quit")
        time.sleep(0.2)
        
    def go_from(self, fen, search_depth = 8):
        
        #pass all out msg first
        while True:
            try:  
                output = self.engine_out_queque.get_nowait()
            except Empty:
                break 
        
        self.send_cmd('position fen ' + fen)
        
        self.last_fen = fen
        
        #if ban_move :
        #        self.send_cmd('banmoves ' + ban_move)
        
        self.send_cmd('go depth  %d' % (search_depth))
        time.sleep(0.2)
        
    def stop_thinking(self):
        self.send_cmd('stop')
        while True:
                try:  
                    output = self.engine_out_queque.get_nowait()
                except Empty:
                    continue
                outputs_list = output.split()
                resp_id = outputs_list[0]
                if resp_id in ['bestmove', 'nobestmove']:         
                        return
        
    def send_cmd(self, cmd_str) :
        
        #print ">>>", cmd_str
        
        try :
            self.pin.write(cmd_str + "\n")
            self.pin.flush()
        except IOError as e :
            print ("error in send cmd", e)
                
    def __handle_engine_out_line(self, output) :
                
        #print "<<<", output
        
        outputs_list = output.split()
        resp_id = outputs_list[0]
        
        if self.enging_status == EngineStatus.BOOTING:
            if resp_id == "id" :
                    self.ids.append(output)
            elif resp_id == "option" :
                    self.options.append(output)
            if resp_id == "ucciok" :
                    self.enging_status = EngineStatus.READY
    
        elif self.enging_status == EngineStatus.READY:
            
            if resp_id == 'nobestmove':         
                print (output)
                self.move_queue.put(("dead", {'fen' : self.last_fen}))
                
            elif resp_id == 'bestmove':
                if outputs_list[1] == 'null':
                    print (output)
                    self.move_queue.put(("dead", {'fen' : self.last_fen}))
                elif outputs_list[-1] == 'draw':
                    self.move_queue.put(("draw", {'fen' : self.last_fen}))
                elif outputs_list[-1] == 'resign':
                    self.move_queue.put(("resign", {'fen' : self.last_fen}))                    
                else :  
                    move_str = output[9:13]
                    pos_move = Move.from_iccs(move_str)
                    
                    move_info = {}    
                    move_info["fen"] = self.last_fen
                    move_info["move"] = pos_move    
                    
                    self.move_queue.put(("best_move",move_info))
                    
            elif resp_id == 'info':
                #info depth 6 score 4 pv b0c2 b9c7 c3c4 h9i7 c2d4 h7e7
                if outputs_list[1] == "depth":
                    move_info = {}    
                    info_list = output[5:].split()
                    
                    if len(info_list) < 5:
                        return
                        
                    move_info["fen"] = self.last_fen
                    move_info[info_list[0]] =  int(info_list[1]) #depth 6
                    move_info[info_list[2]] =  int(info_list[3]) #score 4
                    
                    move_steps = []
                    for step_str in info_list[5:] :
                        move= Move.from_iccs(step_str)
                        move_steps.append(move)    
                    move_info["move"] = move_steps    
                    
                    self.move_queue.put(("info_move", move_info))
    
    def go_best_iccs_move(self, move_str):
    
        pos_move = Move.from_iccs(move_str)
                    
        move_info = {}    
        move_info["fen"] = self.last_fen
        move_info["move"] = pos_move    
    
        self.move_queue.put(("best_move",move_info))
    
                    
#-----------------------------------------------------#

if __name__ == '__main__':
    
    from reader_xqf import *
#    
#    win_dict = { ChessSide.RED : u"红胜", ChessSide.BLACK : u"黑胜" }
#    
#    game = read_from_xqf('test\\ucci_test1.xqf')
#    game.init_board.move_side = ChessSide.RED
#    game.print_init_board()
#    game.print_chinese_moves()
#    
#    board = game.init_board.copy()
#    
#    engine = UcciEngine()
#    engine.load("test\\eleeye\\eleeye.exe")
#    
#    for id in engine.ids:
#        print id
#    for op in engine.options:
#        print op
#    
#    dead = False
#    while not dead:    
#        engine.go_from(board.to_fen(), 10)
#        while True:
#            engine.handle_msg_once()
#            if engine.move_queue.empty():
#                time.sleep(0.2)
#                continue
#            output = engine.move_queue.get()
#            if output[0] == 'best_move':
#                p_from, p_to = output[1]["move"]
#                print board.move(p_from, p_to).to_chinese(),
#                #board.print_board()
#                last_side = board.move_side
#                board.next_turn()
#                break
#            elif output[0] == 'dead':
#                print win_dict[last_side]
#                dead = True
#                break
#            elif output[0] == 'draw':
#                print u'引擎议和'
#                dead = True
#                break
#            elif output[0] == 'resign':
#                print u'引擎认输', win_dict[last_side]
#                dead = True
#                break
#                
#    engine.quit()
#    time.sleep(0.5)    
#