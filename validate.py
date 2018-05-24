from asyncio import Future
import os
import asyncio
from asyncio.queues import Queue
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except:
    print("uvloop not detected, ignoring")
    pass
from cchess_zero import cbf
import tensorflow as tf
import numpy as np
import os
import sys
import random
import time
import argparse
from collections import deque, defaultdict, namedtuple
import scipy.stats
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from cchess_zero.gameboard import *
from net import resnet
import common
from common import board
labels = common.board.create_uci_labels()
from cchess_zero import mcts
from cchess import *
from common import board
import common
from game_convert import boardarr2netinput
uci_labels = common.board.create_uci_labels()
from cchess import BaseChessBoard
from cchess_zero import mcts_pool,mcts_async
from collections import deque, defaultdict, namedtuple
QueueItem = namedtuple("QueueItem", "feature future")
import argparse
import urllib.request
import urllib.parse
from gameplay import GameState,countpiece

parser = argparse.ArgumentParser(description="mcts self play script") 
parser.add_argument('--verbose', '-v', action='store_true', help='verbose mode')
parser.add_argument('--gpu', '-g' , choices=[int(i) for i in list(range(8))],type=int,help="gpu core number",default=0)
parser.add_argument('--server', '-s' ,type=str,help="distributed server location",default=None)
args = parser.parse_args()

gpu_num = int(args.gpu)
server = args.server
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

stamp = time.strftime('%Y-%m-%d',time.localtime(time.time()))
play_save_root = 'data/validate'
gameplay_dir = os.path.join(play_save_root,stamp)
if os.path.exists(gameplay_dir):
    print("dir existed {}".format(gameplay_dir))
else:
    os.mkdir(gameplay_dir)
    print("creating dir {}".format(gameplay_dir))
#(sess,graph),((X,training),(net_softmax,value_head)) 
netold = resnet.get_model('data/prepare_weight/2018-05-23',labels,GPU_CORE=[gpu_num])
netnew = resnet.get_model('data/prepare_weight/2018-05-24_21-19-55',labels,GPU_CORE=[gpu_num])
queue = Queue(400)
async def push_queue( features,loop):
    future = loop.create_future()
    item = QueueItem(features, future)
    await queue.put(item)
    return future
async def prediction_worker(mcts_policy_async,network):
    (sess,graph),((X,training),(net_softmax,value_head)) = network
    q = queue
    while mcts_policy_async.num_proceed < mcts_policy_async._n_playout:
        if q.empty():
            await asyncio.sleep(1e-3)
            continue
        item_list = [q.get_nowait() for _ in range(q.qsize())]
        #print("processing : {} samples".format(len(item_list)))
        features = np.concatenate([item.feature for item in item_list],axis=0)
        
        action_probs, value = sess.run([net_softmax,value_head],feed_dict={X:features,training:False})
        for p, v, item in zip(action_probs, value, item_list):
            item.future.set_result((p, v))
            
async def policy_value_fn_queue(state,loop):
    bb = BaseChessBoard(state.statestr)
    statestr = bb.get_board_arr()
    net_x = np.transpose(boardarr2netinput(statestr,state.get_current_player()),[1,2,0])
    net_x = np.expand_dims(net_x,0)
    future = await push_queue(net_x,loop)
    await future
    policyout,valout = future.result()
    #policyout,valout = sess.run([net_softmax,value_head],feed_dict={X:net_x,training:False})
    #result = work.delay((state.statestr,state.get_current_player()))
    #while True:
    #    if result.ready():
    #        policyout,valout = result.get()
    #        break
    #    else:
    #        await asyncio.sleep(1e-3)
    #policyout,valout = policyout[0],valout[0][0]
    policyout,valout = policyout,valout[0]
    legal_move = GameBoard.get_legal_moves(state.statestr,state.get_current_player())
    #if state.currentplayer == 'b':
    #    legal_move = board.flipped_uci_labels(legal_move)
    legal_move = set(legal_move)
    legal_move_b = set(board.flipped_uci_labels(legal_move))
    
    action_probs = []
    if state.currentplayer == 'b':
        for move,prob in zip(uci_labels,policyout):
            if move in legal_move_b:
                move = board.flipped_uci_labels([move])[0]
                action_probs.append((move,prob))
    else:
        for move,prob in zip(uci_labels,policyout):
            if move in legal_move:
                action_probs.append((move,prob))
    #action_probs = sorted(action_probs,key=lambda x:x[1])
    return action_probs, valout

def get_random_policy(policies):
    sumnum = sum([i[1] for i in policies])
    randnum = random.random() * sumnum
    tmp = 0
    for val,pos in policies:
        tmp += pos
        if tmp > randnum:
            return val

chessplayed = 0
while chessplayed < 20:
    chessplayed += 1
    states = []
    moves = []

    game_states = GameState()
    mcts_policy_w = mcts_async.MCTS(policy_value_fn_queue,n_playout=800,search_threads=16
                                        ,virtual_loss=0.02,policy_loop_arg=True,c_puct=5)
    mcts_policy_b = mcts_async.MCTS(policy_value_fn_queue,n_playout=800,search_threads=16
                                        ,virtual_loss=0.02,policy_loop_arg=True,c_puct=5)
    white_player = 'new'
    black_player = 'old'
    net_white = netnew  
    net_black = netold
    if random.random() < 0.5:
        white_player,black_player = black_player,white_player
        net_white,net_black = net_black,net_white
    result = 'peace'
    can_surrender = False# random.random() > 0.1
    
    peace_round = 0
    remain_piece = countpiece(game_states.statestr)
    for i in range(400):
        begin = time.time()
        is_end,winner = game_states.game_end()
        if is_end == True:
            if winner == -1:
                winner = 'peace'
            result = winner
            break
        start = time.time()
        if i % 2 == 0:
            queue = Queue(400)
            player = 'w'
            if i < 30:
                temp = 1
            else:
                temp = 1e-2
            acts, act_probs = mcts_policy_w.get_move_probs(game_states,temp=temp,verbose=False
                ,predict_workers=[prediction_worker(mcts_policy_w,net_white)])
            policies,score = list(zip(acts, act_probs)),mcts_policy_w._root._Q
            score = -score
            if score < -0.99 and can_surrender:
                winner = 'b'
                break
        else:
            queue = Queue(400)
            player = 'b'

            if i < 14:
                temp = 1
            else:
                temp = 1e-2
            acts, act_probs = mcts_policy_b.get_move_probs(game_states,temp=temp,verbose=False
                ,predict_workers=[prediction_worker(mcts_policy_b,net_black)])
            policies,score = list(zip(acts, act_probs)),mcts_policy_b._root._Q
            if score > 0.99 and can_surrender:
                winner = 'w'
                break

        move = get_random_policy(policies)
        states.append(game_states.statestr)
        moves.append(move)
        game_states.do_move(move)
        
        #peace strategy
        remain_piece_round = countpiece(game_states.statestr)
        if remain_piece_round < remain_piece:
            remain_piece = remain_piece_round
            peace_round = 0
        else:
            peace_round += 1
            
        if i > 150 and peace_round > 60:
            winner = 'peace'
            break
            
        if player == 'w':
            print('{} {} {:.4f}s {:.4f}, sel:{} pol:{} upd:{}'.format(i + 1,move,time.time() - begin,score
                ,mcts_policy_w.select_time,mcts_policy_w.policy_time,mcts_policy_w.update_time))
            mcts_policy_w.select_time,mcts_policy_w.policy_time,mcts_policy_w.update_time = 0,0,0
        else:
            print('{} {} {:.4f}s {:.4f}, sel:{} pol:{} upd:{}'.format(i + 1,move,time.time() - begin,score
                ,mcts_policy_b.select_time,mcts_policy_b.policy_time,mcts_policy_b.update_time))
            mcts_policy_b.select_time,mcts_policy_b.policy_time,mcts_policy_b.update_time = 0,0,0
        mcts_policy_w.update_with_move(move,allow_legacy=False)
        mcts_policy_b.update_with_move(move,allow_legacy=False)
        #print("move {} player {} move {} value {} time {}".format(i + 1,player,move,score,time.time() - start))
    if winner is None:
        winner = 'peace'
    cbfile = cbf.CBF(black='mcts',red='mcts',date='2018-05-113',site='北京',name='noname',datemodify='2018-05-13',
            redteam='icybee',blackteam='icybee',round='第一轮')
    cbfile.receive_moves(moves)
    stamp = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
    randstamp = random.randint(0,1000)

    cbfile.dump('{}/{}_{}_{}-{}_mcts-mcts_{}.cbf'.format(gameplay_dir,stamp,randstamp,white_player,black_player,winner))
mcts_play_wins.append(winner)
