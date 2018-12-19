import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random 
import time
import scipy
import pandas as pd
import tflearn
import copy
import os
import sys
currentpath = os.path.dirname(os.path.realpath(__file__))
project_basedir = os.path.join(currentpath,'..')
sys.path.append(project_basedir)
from gameplays import game_convert
#import convert_game,convert_game_value,convert_game_board

#GPU_CORE = [0]
#BATCH_SIZE = 512
#BEGINING_LR = 0.01
#TESTIMG_WIDTH = 500

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

def res_net_board(inputx,name,training,filters=256,NUM_RES_LAYERS=4):
    net = inputx
    net = tf.layers.conv2d(net,filters=filters,kernel_size=(3,3),activation=None,name="{}_res_convb".format(name),padding='same')
    net = tf.layers.batch_normalization(net,training=training,name="{}_res_bnb".format(name))
    net = tf.nn.elu(net,name="{}_res_elub".format(name))
    for i in range(NUM_RES_LAYERS):
        net = res_block(net,name="{}_layer_{}".format(name,i + 1),training=training,filters=filters)
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

def get_model(MODEL_NAME,labels,GPU_CORE = [0],BATCH_SIZE = 512,NUM_RES_LAYERS = 4,FILTERS = 256,extra = False,extrav2=False):
    tf.reset_default_graph()
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
                if one_core is not None:
                    devicestr = "/gpu:{}".format(one_core) if one_core else ""
                else:
                    devicestr = '/cpu:0'
                with tf.device(devicestr):
                    print(ind)
                    body = res_net_board(X[ind * (BATCH_SIZE // len(GPU_CORE)):(ind + 1) * (BATCH_SIZE // len(GPU_CORE))],
                                         "selectnet",training=training,filters=FILTERS,NUM_RES_LAYERS=NUM_RES_LAYERS)
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
                        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)
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
        #sess.run(tf.global_variables_initializer())

        #tf.train.global_step(sess, global_step)
    if MODEL_NAME is not None:
        with graph.as_default():
            saver = tf.train.Saver(var_list=tf.global_variables())
            saver.restore(sess,MODEL_NAME)
    else:
        with graph.as_default():
            print("initing model weights ....")
            sess.run(tf.global_variables_initializer())
            
    if extrav2:
        return (sess,graph),((X,training),(net_softmax,value_head,train_op_multitarg,(train_op_policy,train_op_value),policy_loss,accuracy_select,global_step,value_loss,nextmove,learning_rate,score,multitarget_loss))
    if extra:
        return (sess,graph),((X,training),(net_softmax,value_head,train_op_multitarg,(train_op_policy,train_op_value),policy_loss,accuracy_select,global_step,value_loss,nextmove,learning_rate,score))
    else:
        return (sess,graph),((X,training),(net_softmax,value_head))
