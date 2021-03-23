import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def fun(
    TIME_STEPS = 20,
    BATCH_SIZE = 50,
    INPUT_SIZE = 13,
    OUTPUT_SIZE = 1,
    CELL_SIZE = 14,
    LSTM_LAYER = 7,
    LR = 0.0006,
    k1=0,
    k2=0,
    N_ITER = 10,
    plot_train_k = 3,
    KEEP_PROB_LSTM = 0.6,
    KEEP_PROB_NN = 0.5,
    NN_LAYER = 2,
    is_train = False,
    is_test = True,
    savingPath = "/Users/ziangcui/Desktop/test/modle.ckpt",
    restorePath = "/Users/ziangcui/Desktop/test/",
    data = pd.read_csv("/Users/ziangcui/Desktop/同花顺/tonghuashun1.csv")):

    tf.reset_default_graph()

    data = np.array(data)
    n1 = len(data[0])-1
    n2 = len(data)

    test_data = data[n2-BATCH_SIZE-TIME_STEPS:, :]
    mean = np.mean(test_data,axis=0)
    std = np.std(test_data, axis=0)

    INPUT_SIZE = n1

    test_data = (test_data - np.mean(test_data, axis=0)) / np.std(test_data, axis=0)

    mydata = []
    mydata_target = []
    for i in range(len(test_data) - TIME_STEPS):
        x = test_data[i:i + TIME_STEPS, :n1]
        y = test_data[i + TIME_STEPS-1, n1]
        mydata.append(x.tolist())
        mydata_target.append(y)



    def get_test_data():
        datat = np.array(mydata)
        return datat

    class LSTMRNN(object):
        def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lstm_layer, nn_layer, lr):
            self.n_steps = n_steps
            self.input_size = input_size
            self.output_size = output_size
            self.cell_size = cell_size
            self.batch_size = batch_size
            self.lstm_layer = lstm_layer
            self.lr = lr
            with tf.name_scope('inputs'):
                self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
                self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys')
                self.keep_prob_lstm = tf.placeholder(tf.float32,name='kpl')
                self.keep_prob_nn = tf.placeholder(tf.float32,name='kpn')
            with tf.variable_scope('in_hidden'):
                self.add_input_layer()
            with tf.variable_scope('LSTM_cell'):
                self.add_cell()
            with tf.variable_scope('out_hidden'):
                self.add_output_layer()
            with tf.variable_scope('out1'):
                self.add_nn_layer(inputs=self.lstm_pred,in_size=self.n_steps,out_size=30,activation_function=tf.nn.relu)
            for i in range(nn_layer-1):
                with tf.variable_scope(str(i)):
                    self.add_nn_layer(inputs=self.pred,in_size=30,out_size=30,activation_function=tf.nn.relu)
            with tf.variable_scope('out3'):
                self.add_nn_layer(inputs=self.pred,in_size=30,out_size=1,activation_function=None)
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

        def add_cell(self):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
            multi_layer_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*self.lstm_layer)
            cell= tf.contrib.rnn.DropoutWrapper(multi_layer_cell,input_keep_prob=self.keep_prob_lstm)
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
                self.lstm_pred = tf.matmul(l_out_x, Ws_out) + bs_out
                self.lstm_pred = tf.reshape(self.lstm_pred,[-1,self.n_steps])

        def add_nn_layer(self,inputs,in_size,out_size,activation_function=None):
            #tf.sqrt(2/in_size)是梯度爆炸/消失的处理方法
            #inputsnn_in = tf.reshape(self.lstm_pred,[-1,in_size])
            Weights = tf.Variable(tf.random_normal([in_size,out_size])*tf.sqrt(2/in_size))
            biases = tf.Variable(tf.zeros([1,out_size])) + 0.1
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
            Wx_plus_b = tf.nn.dropout(Wx_plus_b, self.keep_prob_nn)
            if activation_function == None:
                self.pred = Wx_plus_b
            else:
                self.pred = activation_function(Wx_plus_b)


        def compute_cost(self):

            with tf.name_scope('average_cost'):
                self.cost = tf.reduce_mean(tf.square(tf.reshape(self.pred, [-1]) - tf.reshape(self.ys, [-1])))


        @staticmethod
        def ms_error(labels, logits):
            return tf.square(tf.subtract(labels, logits))

        def _weight_variable(self, shape, name='weights'):
            initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
            return tf.get_variable(shape=shape, initializer=initializer, name=name)

        def _bias_variable(self, shape, name='biases'):
            initializer = tf.constant_initializer(0.1)
            return tf.get_variable(name=name, shape=shape, initializer=initializer)

    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LSTM_LAYER, NN_LAYER, LR)
    sess = tf.Session()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(init)


    if is_test:
        model_file = tf.train.latest_checkpoint(restorePath)
        saver.restore(sess,model_file)
        pred_data = get_test_data()
        feed_dict = {
            model.xs: pred_data,
            model.keep_prob_lstm: 1,
            model.keep_prob_nn: 1
        }

        pred = sess.run(
            model.pred,
            feed_dict=feed_dict)
        pred = pred.reshape((-1))
        pred = pred*std[n1]+mean[n1]
        return pred[BATCH_SIZE-1]