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
def res_block(inputx,name,training,block_num=2,filters=256,kernel_size=(3,3)):
    net = inputx
    for i in range(block_num):
        net = tf.layers.conv2d(net,filters=filters,kernel_size=kernel_size,activation=None,name="{}_res_conv{}".format(name,i),padding='same')
        net = tf.layers.batch_normalization(net,training=training,name="{}_res_bn{}".format(name,i))
        if i == block_num - 1:
            net = net + inputx #= tf.concat((inputx,net),axis=-1)
        net = tf.nn.elu(net,name="{}_res_elu{}".format(name,i))
    return net

def conv_block(inputx,name,training,block_num=1,filters=2,kernel_size=(1,1)):
    net = inputx
    for i in range(block_num):
        net = tf.layers.conv2d(net,filters=filters,kernel_size=kernel_size,activation=None,name="{}_convblock_conv{}".format(name,i),padding='same')
        net = tf.layers.batch_normalization(net,training=training,name="{}_convblock_bn{}".format(name,i))
        net = tf.nn.elu(net,name="{}_convblock_elu{}".format(name,i))
    # net [None,10,9,2]
    netshape = net.get_shape().as_list()
    print("inside conv block {}".format(str(netshape)))
    net = tf.reshape(net,shape=(-1,netshape[1] * netshape[2] * netshape[3]))
    net = tf.layers.dense(net,10 * 9,name="{}_dense".format(name))
    net = tf.nn.elu(net,name="{}_elu".format(name))
    return net

def res_net_board(inputx,name,training,filters=256):
    net = inputx
    net = tf.layers.conv2d(net,filters=filters,kernel_size=(3,3),activation=None,name="{}_res_convb".format(name),padding='same')
    net = tf.layers.batch_normalization(net,training=training,name="{}_res_bnb".format(name))
    net = tf.nn.elu(net,name="{}_res_elub".format(name))
    for i in range(NUM_RES_LAYERS):
        net = res_block(net,name="{}_layer_{}".format(name,i + 1),training=training)
        print(net.get_shape().as_list())
    print("inside res net {}".format(str(net.get_shape().as_list())))
    #net_unsoftmax = conv_block(net,name="{}_conv".format(name),training=training)
    return net

def get_scatter(name):
    with tf.variable_scope("Test"):
        ph = tf.placeholder(tf.float32,name=name)
        op = tf.summary.scalar(name,ph)
    return ph,op


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.


    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads,0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def add_grad_to_list(opt,train_param,loss,tower_grad):
    grads = opt.compute_gradients(loss, var_list = train_param)
    grads = [i[0] for i in grads]
    #print(grads)
    tower_grad.append(zip(grads,train_param))
    
def get_op_mul(tower_gradients,optimizer,gs):
    grads = average_gradients(tower_gradients)
    train_op = optimizer.apply_gradients(grads,gs)
    return train_op

def reduce_mean(x):
    return tf.reduce_mean(x)

def merge(x):
    return tf.concat(x,axis=0)

tf.reset_default_graph()

NUM_RES_LAYERS = 4

graph = tf.Graph()
with graph.as_default():
#with tf.device("/gpu:{}".format(GPU_CORE)):
    X = tf.placeholder(tf.float32,[None,10,9,14])
    nextmove = tf.placeholder(tf.float32,[None,len(labels)])
    score = tf.placeholder(tf.float32,[None,1])
    
    training = tf.placeholder(tf.bool,name='training_mode')
    learning_rate = tf.placeholder(tf.float32)
    global_step = tf.train.get_or_create_global_step()
    optimizer_policy = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
    optimizer_value = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
    optimizer_multitarg = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
    
    tower_gradients_policy,tower_gradients_value,tower_gradients_multitarg = [],[],[]
    
    net_softmax_collection = []
    value_head_collection = []
    multitarget_loss_collection = []
    value_loss_collection = []
    policy_loss_collection = []
    accuracy_select_collection = []
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        for ind,one_core in enumerate(GPU_CORE):
            with tf.device("/gpu:{}".format(one_core)):
                print(ind)
                body = res_net_board(X[ind * (BATCH_SIZE // len(GPU_CORE)):(ind + 1) * (BATCH_SIZE // len(GPU_CORE))],
                                     "selectnet",training=training)
                with tf.variable_scope("policy_head"):
                    policy_head = tf.layers.conv2d(body, 2, 1, padding='SAME')
                    policy_head = tf.contrib.layers.batch_norm(policy_head, center=False, epsilon=1e-5, fused=True,
                                                                is_training=training, activation_fn=tf.nn.relu)

                    # print(self.policy_head.shape)  # (?, 9, 10, 2)
                    policy_head = tf.reshape(policy_head, [-1, 9 * 10 * 2])
                    policy_head = tf.contrib.layers.fully_connected(policy_head, len(labels), activation_fn=None)
                    #self.policy_head.append(policy_head)    # 保存多个gpu的策略头结果（走子概率向量）

                # 价值头
                with tf.variable_scope("value_head"):
                    value_head = tf.layers.conv2d(body, 1, 1, padding='SAME')
                    value_head = tf.contrib.layers.batch_norm(value_head, center=False, epsilon=1e-5, fused=True,
                                                    is_training=training, activation_fn=tf.nn.relu)
                    # print(self.value_head.shape)  # (?, 9, 10, 1)
                    value_head = tf.reshape(value_head, [-1, 9 * 10 * 1])
                    value_head = tf.contrib.layers.fully_connected(value_head, 256, activation_fn=tf.nn.relu)
                    value_head = tf.contrib.layers.fully_connected(value_head, 1, activation_fn=tf.nn.tanh)
                    value_head_collection.append(value_head)
                net_unsoftmax = policy_head

                with tf.variable_scope("Loss"):
                    policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=nextmove[ind * (BATCH_SIZE // len(GPU_CORE)):(ind + 1) * (BATCH_SIZE // len(GPU_CORE))],
                        logits=net_unsoftmax))
                    #loss_summary = tf.summary.scalar("move_loss",policy_loss)
                    value_loss = tf.losses.mean_squared_error(
                        labels=score[ind * (BATCH_SIZE // len(GPU_CORE)):(ind + 1) * (BATCH_SIZE // len(GPU_CORE))],
                        predictions=value_head) 
                    value_loss = tf.reduce_mean(value_loss)
                    regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)
                    regular_variables = tf.trainable_variables()
                    l2_loss = tf.contrib.layers.apply_regularization(regularizer, regular_variables)
                    multitarget_loss = value_loss + policy_loss + l2_loss
                    
                    multitarget_loss_collection.append(multitarget_loss)
                    value_loss_collection.append(value_loss)
                    policy_loss_collection.append(policy_loss)
                net_softmax = tf.nn.softmax(net_unsoftmax)
                net_softmax_collection.append(net_softmax)
                
                correct_prediction = tf.equal(tf.argmax(nextmove,1), tf.argmax(net_softmax,1))

                with tf.variable_scope("Accuracy"):
                    accuracy_select = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    accuracy_select_collection.append(accuracy_select)
                tf.get_variable_scope().reuse_variables()
                trainable_params = tf.trainable_variables()
                tp_policy = [i for i in trainable_params if 
                                    ('value_head' not in i.name)]
                tp_value = [i for i in trainable_params if 
                                    ('policy_head' not in i.name)]

                add_grad_to_list(optimizer_policy,tp_policy,policy_loss,tower_gradients_policy)
                add_grad_to_list(optimizer_value,tp_value,value_loss,tower_gradients_value)
                add_grad_to_list(optimizer_multitarg,trainable_params,multitarget_loss,tower_gradients_multitarg)
               
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        #gradients_policy = average_gradients(tower_gradients_policy)
        train_op_policy = get_op_mul(tower_gradients_policy,optimizer_policy,global_step)
        train_op_value = get_op_mul(tower_gradients_value,optimizer_value,global_step)
        train_op_multitarg = get_op_mul(tower_gradients_multitarg,optimizer_multitarg,global_step)
        #train_op = optimizer.minimize(policy_loss,global_step=global_step)
    net_softmax = merge(net_softmax_collection)
    value_head = merge(value_head_collection)
    multitarget_loss = reduce_mean(multitarget_loss_collection)
    value_loss = reduce_mean(value_loss_collection)
    policy_loss = reduce_mean(policy_loss_collection)
    accuracy_select = reduce_mean(accuracy_select_collection)
    
with graph.as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    tf.train.global_step(sess, global_step)
    
import os
if not os.path.exists("models/{}".format(model_name)):
    os.mkdir("models/{}".format(model_name))
    
N_BATCH = int(len(trainset) / BATCH_SIZE) * 40

latest_netname = NetMatainer(None,network_dir).get_latest()

with graph.as_default():
    train_epoch = 30
    train_batch = 0
    saver = tf.train.Saver(var_list=tf.global_variables())
    saver.restore(sess,os.path.join(network_dir,latest_netname))
    
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