import os
import sys
currentpath = os.path.dirname(os.path.realpath(__file__))
project_basedir = os.path.join(currentpath,'..')
sys.path.append(project_basedir)
from gameplays import players,gameplay
from net import resnet
import common
import time
import numpy as np
import random
from cchess_zero import cbf
from config import conf
import os
from net import net_maintainer
import urllib
import tensorflow as tf
from net.resnet import get_model

network = resnet.get_model(None,common.board.create_uci_labels(),GPU_CORE=[""],FILTERS=conf.network_filters,NUM_RES_LAYERS=conf.network_layers)
(sess,graph),_ = network

stamp = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
network_dir = conf.distributed_server_weight_dir
dst = os.path.join(network_dir,"{}".format(stamp))

with graph.as_default():
    saver = tf.train.Saver(var_list=tf.global_variables())
    saver.save(sess,dst)
