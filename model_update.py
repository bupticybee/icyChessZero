import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random 
import time
from utils import Dataset,ProgressBar
from tflearn.data_flow import DataFlow,DataFlowStatus,FeedDictFlow
from tflearn.data_utils import Preloader,ImagePreloader
import scipy
import pandas as pd
import xmltodict
import common
import tflearn
import copy
from cchess import *
from game_convert import convert_game,convert_game_value,convert_game_board
import os, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from net.net_maintainer import NetMatainer
from net import resnet

stamp = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))

data_dir = os.path.join('data/history_selfplays',stamp)
if os.path.exists(data_dir):
    print('data_dir already exist: {}'.format(data_dir))
else:
    print('creating data_dir: {}'.format(data_dir))
    os.mkdir("{}".format(data_dir))
    
GPU_CORE = [0]
BATCH_SIZE = 512
BEGINING_LR = 0.01
#TESTIMG_WIDTH = 500
model_name = 'update_model'

distribute_dir = 'data/distributed/'
filelist = os.listdir(distribute_dir)

network_dir = 'data/prepare_weight/'

for f in filelist:
    src = os.path.join(distribute_dir,f)
    dst = os.path.join(data_dir,f)
    shutil.move(src,dst)
    
filelist = [os.path.join(data_dir,i) for i in filelist]

labels = common.board.create_uci_labels()
label2ind = dict(zip(labels,list(range(len(labels)))))

rev_ab = dict(zip('abcdefghi','abcdefghi'[::-1]))
rev_num = dict(zip('0123456789','0123456789'[::-1]))


class ElePreloader(object):
    def __init__(self,filelist,batch_size=64):
        self.batch_size=batch_size
        #content = pd.read_csv(datafile,header=None,index_col=None)
        self.filelist = filelist#[i[0] for i in content.get_values()]
        self.pos = 0
        self.feature_list = {"red":['A', 'B', 'C', 'K', 'N', 'P', 'R']
                             ,"black":['a', 'b', 'c', 'k', 'n', 'p', 'r']}
        self.batch_size = batch_size
        self.batch_iter = self.__iter()
        assert(len(self.filelist) > batch_size)
        self.game_iterlist = [None for i in self.filelist]
    
    def __iter(self):
        retx1,rety1,retx2,rety2 = [],[],[],[]
        vals = []
        filelist = []
        while True:
            for i in range(self.batch_size):
                if self.game_iterlist[i] == None:
                    if len(filelist) == 0:
                        filelist = copy.copy(self.filelist)
                        random.shuffle(filelist)
                    self.game_iterlist[i] = convert_game_value(filelist.pop(),self.feature_list,None)
                game_iter = self.game_iterlist[i]
                
                try:
                    x1,y1,val1 = game_iter.__next__()
                    x1 = np.transpose(x1,[1,2,0])
                    x1 = np.expand_dims(x1,axis=0)
                    
                    if random.random() < 0.5:
                        y1 = [rev_ab[y1[0]],y1[1],rev_ab[y1[2]],y1[3]]
                        x1 = x1[:,:,::-1,:]
                        #x1 = np.concatenate((x1[:,::-1,:,7:],x1[:,::-1,:,:7]),axis=-1)
                    retx1.append(x1)
                    #rety1.append(y1)
                    oney = np.zeros(len(labels))
                    oney[label2ind[''.join(y1)]] = 1
                    rety1.append(oney)
                    vals.append(val1)

                    if len(retx1) >= self.batch_size:
                        yield (np.concatenate(retx1,axis=0),np.asarray(rety1),np.asarray(vals))
                        retx1,rety1 = [],[]
                        vals = []
                except :
                    self.game_iterlist[i] = None

    def __getitem__(self, id):
        
        x1,y1,val1 = self.batch_iter.__next__()
        return x1,y1,val1
        
    def __len__(self):
        return len(self.filelist)
    
trainset = ElePreloader(filelist=filelist,batch_size=BATCH_SIZE)
with tf.device("/gpu:{}".format(GPU_CORE[0])):
    coord = tf.train.Coordinator()
    trainflow = FeedDictFlow({
            'data':trainset,
        },coord,batch_size=BATCH_SIZE,shuffle=True,continuous=True,num_threads=1)
trainflow.start()
sample_x1,sample_y1,sample_value = trainflow.next()['data']

print(sample_x1.shape,sample_y1.shape,sample_value.shape)


    
import os
if not os.path.exists("models/{}".format(model_name)):
    os.mkdir("models/{}".format(model_name))
    
N_BATCH = int(len(trainset) / BATCH_SIZE) * 75 

latest_model_name = NetMatainer(None,network_dir).get_latest()
(sess,graph),((X,training),(net_softmax,value_head,train_op_policy,train_op_value,policy_loss,accuracy_select,global_step,value_loss,nextmove,learning_rate,score)) = resnet.get_model(os.path.join(network_dir,latest_model_name),labels,GPU_CORE=GPU_CORE,FILTERS=128,NUM_RES_LAYERS=7,extra=True)
    
train_epoch = 1
train_batch = 0

restore = True
N_EPOCH = 3
DECAY_EPOCH = 20

class ExpVal:
    def __init__(self,exp_a=0.97):
        self.val = None
        self.exp_a = exp_a
    def update(self,newval):
        if self.val == None:
            self.val = newval
        else:
            self.val = self.exp_a * self.val + (1 - self.exp_a) * newval
    def getval(self):
        return round(self.val,2)
    
expacc_move = ExpVal()
exploss = ExpVal()
expsteploss = ExpVal()

begining_learning_rate = 1e-2

pred_image = None
if restore == False:
    train_epoch = 1
    train_batch = 0
for one_epoch in range(train_epoch,N_EPOCH):
    train_epoch = one_epoch
    pb = ProgressBar(worksum=N_BATCH * BATCH_SIZE,info=" epoch {} batch {}".format(train_epoch,train_batch))
    pb.startjob()
    
    for one_batch in range(N_BATCH):
        if restore == True and one_batch < train_batch:
            pb.auto_display = False
            pb.complete(BATCH_SIZE)
            pb.auto_display = True
            continue
        else:
            restore = False
        train_batch = one_batch
        
        batch_x,batch_y,batch_v = trainflow.next()['data']
        batch_v = np.expand_dims(np.nan_to_num(batch_v),1)
        # learning rate decay strategy
        batch_lr = begining_learning_rate * 2 ** -(one_epoch // DECAY_EPOCH)
        with graph.as_default():
            _,step_loss,step_acc_move,step_value = sess.run(
                [train_op_policy,policy_loss,accuracy_select,global_step],feed_dict={
                    X:batch_x,nextmove:batch_y,learning_rate:batch_lr,training:True,
                })
            _,step_value_loss,step_val_predict = sess.run(
                [train_op_value,value_loss,value_head],feed_dict={
                    X:batch_x,learning_rate:batch_lr,training:True,score:batch_v,
                })
            batch_v = - batch_v
            batch_x = np.concatenate((batch_x[:,::-1,:,7:],batch_x[:,::-1,:,:7]),axis=-1)
            _,step_value_loss,step_val_predict = sess.run(
                [train_op_value,value_loss,value_head],feed_dict={
                    X:batch_x,learning_rate:batch_lr,training:True,score:batch_v,
                })
            
        
        step_acc_move *= 100
        
        expacc_move.update(step_acc_move)
        exploss.update(step_loss)
        expsteploss.update(step_value_loss)

       
        pb.info = "EPOCH {} STEP {} LR {} ACC {} LOSS {} value_loss {}".format(
            one_epoch,one_batch,batch_lr,expacc_move.getval(),exploss.getval(),expsteploss.getval())
        
        pb.complete(BATCH_SIZE)
    print()
    with graph.as_default():
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.save(sess,"models/{}/model_{}".format(model_name,one_epoch))
        
for f in ['data-00000-of-00001','meta','index']:
    src = "models/{}/model_{}.{}".format(model_name,one_epoch,f)
    dst = os.path.join(network_dir,"{}.{}".format(stamp,f))
    shutil.copyfile(src,dst)
