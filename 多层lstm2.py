#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:41:54 2020

@author: ziangcui
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from datetime import datetime

tf.reset_default_graph()

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "/Users/ziangcui/Desktop/log"
logdir = "{}/run-{}/".format(root_logdir, now)

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
OUTPUT_SIZE = 1
CELL_SIZE = 14
LSTM_LAYER = 2
LR = 0.0006
k1 = 0
k2 = 0
N_ITER = 5
plot_train_k = 3
KEEP_PROB_LSTM = 0.6
KEEP_PROB_NN = 1

is_train = True
is_test = True

savingPath = "test/modle.ckpt"
restorePath = "test/"

data = pd.read_csv("/Users/ziangcui/Desktop/同花顺/tonghuashun1.csv")
data = np.array(data)
n1 = len(data[0]) - 1
train_end_index = math.floor(len(data) * 0.9)
data_train = data[0:train_end_index]
data_test = data[train_end_index + 1:]

mean_train = np.mean(data_train, axis=0)
std_train = np.std(data_train, axis=0)
mean_test = np.mean(data_test, axis=0)
std_test = np.std(data_test, axis=0)

INPUT_SIZE = n1

data_train = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)
data_test = (data_test - np.mean(data_test, axis=0)) / np.std(data_test, axis=0)

train_data = []
train_target = []

for i in range(len(data_train) - TIME_STEPS):
    x = data[i:i + TIME_STEPS, :n1]
    # y存储label
    y = data[i:i + TIME_STEPS, n1, np.newaxis]
    # np.newaxis分别是在行或列上增加维度
    train_data.append(x.tolist())
    train_target.append(y.tolist())

test_data = []
test_target = []

test_size = (len(data_test) + TIME_STEPS - 1) // TIME_STEPS
for i in range(test_size - 1):
    x = data_test[i * TIME_STEPS:(i+1) * TIME_STEPS, :n1]
    y = data_test[i * TIME_STEPS:(i+1) * TIME_STEPS, n1]
    test_data.append(x.tolist())
    test_target.append(y)


def get_train_pic():
    k = len(train_data) // BATCH_SIZE - plot_train_k
    datat = data_train[k * BATCH_SIZE:(k + plot_train_k) * BATCH_SIZE, n1]
    return datat


def get_batch():
    seq1 = np.array(train_data)
    seq1 = seq1[k1:k1 + BATCH_SIZE, :, :]
    res1 = np.array(train_target)
    res1 = res1[k1:k1 + BATCH_SIZE, :, :]
    return [seq1, res1]


def get_test_data():
    datat = np.array(test_data)
    datat = datat[k2:k2 + BATCH_SIZE, :, :]
    return [datat, test_target]


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lstm_layer):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.lstm_layer = lstm_layer
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
            self.keep_prob_lstm = tf.placeholder(tf.float32, name='kpl')
            self.keep_prob_nn = tf.placeholder(tf.float32, name='kpn')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self, ):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.lstm_layer)
        cell = tf.contrib.rnn.DropoutWrapper(multi_layer_cell, input_keep_prob=self.keep_prob_lstm)
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
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LSTM_LAYER)
    sess = tf.Session()
    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(init)

    if is_train:
        t = 0
        max_cost = 10000
        for iteration in range(N_ITER):
            k1 = 0
            for i in range(len(train_data) // BATCH_SIZE):
                seq, res = get_batch()
                k1 = k1 + BATCH_SIZE
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.keep_prob_lstm: KEEP_PROB_LSTM,
                    model.keep_prob_nn: KEEP_PROB_NN
                    # create initial state
                }

                _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict)

            if iteration % 20 == 0:
                print('cost out: ', cost)
                result = sess.run(merged, feed_dict)
                # writer.add_summary(result, i)
            if cost < max_cost:
                max_cost = cost
                save_path = saver.save(sess, savingPath)

        # writer.close()
        print("**********************")

    if is_test:
        model_file = tf.train.latest_checkpoint(restorePath)
        saver.restore(sess, model_file)

        test_predict = []
        train_pred = []

        k1 = 0
        for i in range(len(train_data) // BATCH_SIZE):
            if i >= (len(train_data) // BATCH_SIZE - plot_train_k):
                seq, res = get_batch()

                feed_dict = {
                    model.xs: seq,
                    model.keep_prob_lstm: 1,
                    model.keep_prob_nn: 1
                }
                pred = sess.run(
                    model.pred,
                    feed_dict=feed_dict)

                for j in range(BATCH_SIZE*TIME_STEPS):
                    if j%20==0:
                        train_pred.append(pred[j])
            k1 = k1 + BATCH_SIZE

        test_x, test_y = get_test_data()

        for step in range(len(test_x)-1):
            feed_dict = {
                model.xs: [test_x[step]],
                model.keep_prob_lstm: 1,
                model.keep_prob_nn: 1
            }
            pred1 = sess.run(
                model.pred,
                feed_dict=feed_dict)

            pred1 = pred1.reshape((-1))
            test_predict.extent(pred1)

        #test_y = test_y[:(len(test_data) // BATCH_SIZE) * BATCH_SIZE]
        print("finish")

        train_real = get_train_pic()
        train_real = np.array(train_real) * std_train[n1] + mean_train[n1]
        train_pred = np.array(train_pred) * std_train[n1] + mean_train[n1]
        # 相对误差=（测量值-计算值）/计算值×100%
        acc = np.average(np.abs(train_pred - train_real[:len(train_pred)]) / train_real[:len(train_pred)])
        print("训练预测的相对误差:", acc)

        test_y = np.array(test_y) * std_test[n1] + mean_test[n1]
        test_predict = np.array(test_predict) * std_test[n1] + mean_test[n1]

        test_y = test_y.reshape(-1, 1)
        test_predict = test_predict.reshape(-1, 1)
        train_real = train_real.reshape(-1, 1)
        train_pred = train_pred.reshape(-1, 1)

        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])
        print("测试预测的相对误差:", acc)

        test_predict = np.vstack((train_pred, test_predict))
        test_y = np.vstack((train_real, test_y))

        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])
        print("总的相对误差:", acc)

        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b', )
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.xlabel('time value/day', fontsize=14)
        plt.ylabel('close value/point', fontsize=14)
        plt.title('predict-----blue,real-----red', fontsize=10)
        plt.show()
