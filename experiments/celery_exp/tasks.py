from celery import Celery
from asycele.batches import Batches
import time
import asyncio
#import threading
import multiprocessing
import os
from celery.result import allow_join_result
from celery import group
import sys
sys.path.append('../')
import numpy as np
from celery.utils.log import get_task_logger
from celery.signals import worker_init, worker_process_init
from net import resnet
logger = get_task_logger(__name__)
#lock = multiprocessing.Lock()
import common
from cchess import BaseChessBoard
from game_convert import boardarr2netinput

tf_model = None

app = Celery('tasks_chess',backend='redis://localhost',broker='redis://localhost')
#app.conf.task_serializer   = 'json'
#app.conf.result_serializer = 'json'

@worker_process_init.connect()
def on_worker_init(**_):
    global tf_model
    # Create server with model
    logger.info('model for worker: started init')
    labels = common.board.create_uci_labels()
    tf_model = resnet.get_model('models/5_7_resnet_joint-two_stage/model_57',labels)
    logger.info('model for worker: initialized')

def proc(x):
    statestr,player = x
    bb = BaseChessBoard(statestr)
    statestr = bb.get_board_arr()
    net_x = np.transpose(boardarr2netinput(statestr,player),[1,2,0])
    net_x = np.expand_dims(net_x,0)
    return net_x
    

def real_work(requests):
    (sess,graph),((X,training),(net_softmax,value_head)) = tf_model
    #with lock:
    #print("in: {}".format(os.getpid()))
    #print(len(requests))
    #proc = lambda x:x
    in_arr = [proc(*request.args,**request.kwargs) for request in requests]
    in_arr = np.concatenate(in_arr,axis=0)
    policyout,valout = sess.run([net_softmax,value_head],feed_dict={X:in_arr,training:False})
    
    for one_pol,one_val,request in zip(policyout,valout,requests):
        app.backend.mark_as_done(request.id,(one_pol.tolist(),float(one_val[0])))

@app.task(base=Batches,flush_every=256,flush_interval=0.00001,ignore_result = True)
def work(requests):
    with allow_join_result():
        real_work(requests)
