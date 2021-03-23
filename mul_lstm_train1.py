#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:57:52 2020

@author: ziangcui
"""


import tensorflow as tf
import numpy as np
import math



def fun(TIME_STEPS,
    BATCH_SIZE,
    CELL_SIZE,
    LSTM_LAYER,
    LR,
    N_ITER,
    KEEP_PROB_LSTM,
    is_train,
    savingPath,
    data):

    tf.reset_default_graph()
    OUTPUT_SIZE = 1
    k1 = 0
    data = np.array(data)

    index = []
    for i in range(len(data)):
        if 0 in data[i]:
            index.append(i)
    for i in range(len(index)):
        data = np.delete(data, index[len(index) - i - 1], 0)

    n1 = len(data[0])-1
    train_end_index = math.floor(len(data)*0.8)
    train_end_index1 = math.floor(len(data) * 0.5)
    data_train = data[0:train_end_index]

    INPUT_SIZE = n1

    train_data = []
    train_target = []

    data_train = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)

    data_test = data[train_end_index1+1:]

    data_test = data_test[len(data_test) - len(data_test)//(TIME_STEPS*BATCH_SIZE)*(TIME_STEPS*BATCH_SIZE):]
    mean_test = np.mean(data_test,axis=0)
    std_test = np.std(data_test,axis=0)
    data_test = (data_test - mean_test) / std_test

    test_data = []
    test_target = data_test[:, n1]

    for i in range(len(data_test)//TIME_STEPS):
        x = data_test[i * TIME_STEPS:(i+1) * TIME_STEPS, :n1]
        test_data.append(x.tolist())

    test_data = np.array(test_data)


    for i in range(len(data_train) - TIME_STEPS):
        x = data_train[i:i + TIME_STEPS, :n1]
        y = data_train[i:i + TIME_STEPS, n1, np.newaxis]
        train_data.append(x.tolist())
        train_target.append(y.tolist())


    def get_batch():
        seq1 = np.array(train_data)
        seq1 = seq1[k1*BATCH_SIZE:(k1+1)*BATCH_SIZE,:,:]
        res1 = np.array(train_target)
        res1 = res1[k1*BATCH_SIZE:(k1+1)*BATCH_SIZE,:,:]
        return [seq1, res1]

    class LSTMRNN(object):
        def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lstm_layer, lr):
            self.n_steps = n_steps
            self.input_size = input_size
            self.output_size = output_size
            self.cell_size = cell_size
            self.batch_size = batch_size
            self.lstm_layer = lstm_layer
            self.lr = lr
            with tf.name_scope('inputs'):
                self.xs = tf.placeholder(tf.float32, [None, n_steps, self.input_size], name='xs')
                self.ys = tf.placeholder(tf.float32, [None, n_steps, self.output_size], name='ys')
                self.keep_prob_lstm = tf.placeholder(tf.float32,name='kpl')
            with tf.variable_scope('in_hidden'):
                self.add_input_layer()
            with tf.variable_scope('LSTM_cell'):
                self.add_cell()
            with tf.variable_scope('out_hidden'):
                self.add_output_layer()
            with tf.name_scope('cost'):
                self.compute_cost()
            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        def add_input_layer(self,):
            l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
            # Ws (in_size, cell_size)
            Ws_in = self._weight_variable([self.input_size, self.cell_size])
            # bs (cell_size, )
            bs_in = self._bias_variable([self.cell_size,])
            # l_in_y = (batch * n_steps, cell_size)
            with tf.name_scope('Wx_plus_b'):
                l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
            # reshape l_in_y ==> (batch, n_steps, cell_size)
            self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

        # def add_cell(self):
        #     lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=0.7, state_is_tuple=True)
        #     multi_layer_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*self.lstm_layer)
        #     cell= tf.contrib.rnn.DropoutWrapper(multi_layer_cell,input_keep_prob=self.keep_prob_lstm)
        #     with tf.name_scope('initial_state'):
        #         self.cell_init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        #     self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
        #         cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

        def add_cell(self):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
            d_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob_lstm)
            cell = tf.nn.rnn_cell.MultiRNNCell([d_cell]*self.lstm_layer)
            with tf.name_scope('initial_state'):
                self.cell_init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

        def add_output_layer(self):
            # shape = (batch * steps, cell_size)
            l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
            Ws_out = self._weight_variable([self.cell_size, self.output_size])
            bs_out = self._bias_variable([self.output_size, ])
            # shape = (batch * steps, output_size)
            with tf.name_scope('Wx_plus_b'):
                self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

        def compute_cost(self):

            with tf.name_scope('average_cost'):
                self.cost = tf.reduce_mean(tf.square(tf.reshape(self.pred, [-1]) - tf.reshape(self.ys, [-1])))

        def _weight_variable(self, shape, name='weights'):
            initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
            return tf.get_variable(shape=shape, initializer=initializer, name=name)

        def _bias_variable(self, shape, name='biases'):
            initializer = tf.constant_initializer(0.1)
            return tf.get_variable(name=name, shape=shape, initializer=initializer)

    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LSTM_LAYER, LR)
    sess = tf.Session()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(init)

    if is_train:
        cost_sum = []
        acc_min = 1
        for iteration in range(N_ITER):
            k1=0
            for i in range(len(train_data)//BATCH_SIZE):
                seq, res = get_batch()
                k1 = k1 + 1
                if iteration == 0:
                    feed_dict = {
                        model.xs: seq,
                        model.ys: res,
                        model.keep_prob_lstm:KEEP_PROB_LSTM
                        # create initial state
                    }
                else:
                    feed_dict = {
                        model.xs: seq,
                        model.ys: res,
                        model.keep_prob_lstm:KEEP_PROB_LSTM,
                        model.cell_init_state: state    # use last state as the initial state for this run
                    }

                _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict)

            cost_sum.append(cost)

            if iteration%5==0:
                print('cost out: ', cost)
            if iteration%5==0:
                test_predict = []
                for step in range(len(test_data) // BATCH_SIZE):
                    feed_dict = {
                        model.xs: test_data[step * BATCH_SIZE:(step + 1) * BATCH_SIZE, :, :],
                        model.keep_prob_lstm: 1,
                    }
                    pred = sess.run(
                        model.pred,
                        feed_dict=feed_dict)
                    pred = pred.reshape((-1))
                    test_predict.extend(pred)

                test_target = np.array(test_target) * std_test[n1] + mean_test[n1]
                test_predict = np.array(test_predict) * std_test[n1] + mean_test[n1]

                test_target = test_target.reshape(-1, 1)
                test_predict = test_predict.reshape(-1, 1)

                acc = np.average(np.abs(test_predict - test_target[:len(test_predict)]) / test_target[:len(test_predict)])
                test_target = (np.array(test_target) - mean_test[n1]) / std_test[n1]
                print(acc)

            if acc < acc_min:
                acc_min = acc
                save_path = saver.save(sess,savingPath)
        print(acc_min)


        print("***************")
        return cost_sum


