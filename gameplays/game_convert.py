import xmltodict
import sys
import os
currentpath = os.path.dirname(os.path.realpath(__file__))
project_basedir = os.path.join(currentpath,'..')
sys.path.append(project_basedir)
from cchess import *
from common import board

def boardarr2netinput(boardarr,player,feature_list={"black":['A', 'B', 'C', 'K', 'N', 'P', 'R']
                             ,"red":['a', 'b', 'c', 'k', 'n', 'p', 'r']}):
    # chess picker features
    picker_x = []
    #picker_y = []
    if player == 'b':
        for one in feature_list['red']:
            picker_x.append(np.asarray(boardarr == one,dtype=np.uint8))
        for one in feature_list['black']:
            picker_x.append(np.asarray(boardarr == one,dtype=np.uint8))
    elif player == 'w':
        for one in feature_list['black']:
            picker_x.append(np.asarray(boardarr == one,dtype=np.uint8))
        for one in feature_list['red']:
            picker_x.append(np.asarray(boardarr == one,dtype=np.uint8))
    picker_x = np.asarray(picker_x)
    
    if player == 'b':
        return picker_x
    elif player == 'w':
        return picker_x[:,::-1,:]

def convert_game_board(onefile,feature_list,pgn2value):
    doc = xmltodict.parse(open(onefile,encoding='utf-8').read())
    fen = doc['ChineseChessRecord']["Head"]["FEN"]
    pgnfile = doc['ChineseChessRecord']["Head"]["From"]
    moves = [i["@value"] for i in  doc['ChineseChessRecord']['MoveList']["Move"] if i["@value"] != '00-00']
    bb = BaseChessBoard(fen)
    val = pgn2value[pgnfile]
    red = False
    for i in moves:
        red = not red
        x1,y1,x2,y2 = int(i[0]),int(i[1]),int(i[3]),int(i[4])
        #print("{} {}".format(i,"红" if red else "黑"))
        
        boardarr = bb.get_board_arr()
        
        # chess picker features
        picker_x = []
        #picker_y = []
        yield bb,(x1,y1,x2,y2)
        moveresult = bb.move(Pos(x1,y1),Pos(x2,y2))
        assert(moveresult != None)

def is_game_valid(onefile,feature_list,pgn2value):
    doc = xmltodict.parse(open(onefile,encoding='utf-8').read())
    fen = doc['ChineseChessRecord']["Head"]["FEN"]
    if pgn2value is not None:
        pgnfile = doc['ChineseChessRecord']["Head"]["From"]
    #moves = [i["@value"] for i in  doc['ChineseChessRecord']['MoveList']["Move"] if i["@value"] != '00-00']
    bb = BaseChessBoard(fen)
    if pgn2value is not None:      
        if pgnfile not in pgn2value:
            return False
        else:
            return True
        
def convert_game_value(onefile,feature_list,pgn2value):
    doc = xmltodict.parse(open(onefile,encoding='utf-8').read())
    fen = doc['ChineseChessRecord']["Head"]["FEN"]
    if pgn2value is not None:
        pgnfile = doc['ChineseChessRecord']["Head"]["From"]
    moves = [i["@value"] for i in  doc['ChineseChessRecord']['MoveList']["Move"] if i["@value"] != '00-00']
    bb = BaseChessBoard(fen)
    if pgn2value is not None:      
        val = pgn2value[pgnfile]
        #print(val)
    else:
        place = onefile.split('.')[-2].split('_')[-1]
        if place == 'w':
            val = 1
        elif place == 'b':
            val = -1
        else:
            val = 0
    red = False
    for i in moves:
        red = not red
        x1,y1,x2,y2 = int(i[0]),int(i[1]),int(i[3]),int(i[4])
        #print("{} {}".format(i,"红" if red else "黑"))
        
        boardarr = bb.get_board_arr()
        
        # chess picker features
        picker_x = []
        #picker_y = []
        if red:
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
        picker_y = np.asarray([
            board.x_axis[x1],
            board.y_axis[y1],
            board.x_axis[x2],
            board.y_axis[y2],
            ])
        picker_y_rev = np.asarray([
            board.x_axis[x1],
            board.y_axis[9 - y1],
            board.x_axis[x2],
            board.y_axis[9 - y2],
            ])
        #target = np.zeros((10,9))
        #target[y1,x1] = 1
        #picker_y = target
        #
        ## chess mover features
        #mover_x = []
        #mover_y = []
        #mover_x = np.concatenate((picker_x,target.reshape((1,10,9))))
        #mover_y = np.zeros((10,9))
        #mover_y[y2,x2] = 1
        if red:
            yield picker_x,picker_y,val#,picker_y,mover_x,mover_y
        else:
            yield picker_x[:,::-1,:],picker_y_rev,-val#,picker_y[::-1,:],mover_x[:,::-1,:],mover_y[::-1,:]
        moveresult = bb.move(Pos(x1,y1),Pos(x2,y2))
        assert(moveresult != None)
        
def convert_game(onefile,feature_list):
    doc = xmltodict.parse(open(onefile,encoding='utf-8').read())
    fen = doc['ChineseChessRecord']["Head"]["FEN"]
    pgnfile = doc['ChineseChessRecord']["Head"]["From"]
    moves = [i["@value"] for i in  doc['ChineseChessRecord']['MoveList']["Move"] if i["@value"] != '00-00']
    bb = BaseChessBoard(fen)
    red = False
    for i in moves:
        red = not red
        x1,y1,x2,y2 = int(i[0]),int(i[1]),int(i[3]),int(i[4])
        #print("{} {}".format(i,"红" if red else "黑"))
        
        boardarr = bb.get_board_arr()
        
        # chess picker features
        picker_x = []
        #picker_y = []
        if red:
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
        picker_y = np.asarray([
            board.x_axis[x1],
            board.y_axis[y1],
            board.x_axis[x2],
            board.y_axis[y2],
            ])
        picker_y_rev = np.asarray([
            board.x_axis[x1],
            board.y_axis[9 - y1],
            board.x_axis[x2],
            board.y_axis[9 - y2],
            ])
        #target = np.zeros((10,9))
        #target[y1,x1] = 1
        #picker_y = target
        #
        ## chess mover features
        #mover_x = []
        #mover_y = []
        #mover_x = np.concatenate((picker_x,target.reshape((1,10,9))))
        #mover_y = np.zeros((10,9))
        #mover_y[y2,x2] = 1
        if red:
            yield picker_x,picker_y#,picker_y,mover_x,mover_y
        else:
            yield picker_x[:,::-1,:],picker_y_rev#,picker_y[::-1,:],mover_x[:,::-1,:],mover_y[::-1,:]
        moveresult = bb.move(Pos(x1,y1),Pos(x2,y2))
        assert(moveresult != None)
        
def convert_value(onefile,feature_list):
    doc = xmltodict.parse(open(onefile,encoding='utf-8').read())
    fen = doc['ChineseChessRecord']["Head"]["FEN"]
    pgnfile = doc['ChineseChessRecord']["Head"]["From"]
    moves = [i["@value"] for i in  doc['ChineseChessRecord']['MoveList']["Move"] if i["@value"] != '00-00']
    bb = BaseChessBoard(fen)
    red = False
    for i in moves:
        red = not red
        x1,y1,x2,y2 = int(i[0]),int(i[1]),int(i[3]),int(i[4])
        #print("{} {}".format(i,"红" if red else "黑"))
        
        boardarr = bb.get_board_arr()
        
        # chess picker features
        picker_x = []
        picker_y = []
        if red:
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

        
        picker_y = target
        

        if red:
            yield picker_x,
        else:
            yield picker_x[:,::-1,:],
        assert(moveresult != None)
