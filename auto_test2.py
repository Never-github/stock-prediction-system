import tensorflow as tf
import numpy as np
import math
import ray
from ray import tune
import parameter_list as para

def auto_func(xdata):
    tf.reset_default_graph()

    ray.init()

    class LSTMRNN(object):
        def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lstm_layer, nn_layer, lr,
                     nn_sellsize, lstm_output):
            self.n_steps = n_steps
            self.input_size = input_size
            self.output_size = output_size
            self.cell_size = cell_size
            self.batch_size = batch_size
            self.lstm_layer = lstm_layer
            self.nn_cellsize = nn_sellsize
            self.lstm_output = lstm_output
            self.lr = lr
            with tf.name_scope('inputs'):
                self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
                self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys')
                self.keep_prob_lstm = tf.placeholder(tf.float32, name='kpl')
                self.keep_prob_nn = tf.placeholder(tf.float32, name='kpn')
            with tf.variable_scope('in_hidden'):
                self.add_input_layer()
            with tf.variable_scope('LSTM_cell'):
                self.add_cell()
            with tf.variable_scope('out_hidden'):
                self.add_output_layer()
            with tf.variable_scope('out1'):
                self.add_nn_layer(inputs=self.lstm_pred, in_size=self.n_steps * self.lstm_output,
                                  out_size=self.nn_cellsize, activation_function=tf.nn.relu)
            for i in range(nn_layer - 1):
                with tf.variable_scope(str(i)):
                    self.add_nn_layer(inputs=self.pred, in_size=self.nn_cellsize, out_size=self.nn_cellsize,
                                      activation_function=tf.nn.relu)
            with tf.variable_scope('out3'):
                self.add_nn_layer(inputs=self.pred, in_size=self.nn_cellsize, out_size=1, activation_function=None)
            with tf.name_scope('cost'):
                self.compute_cost()
            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

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
            Ws_out = self._weight_variable([self.cell_size, self.lstm_output])
            bs_out = self._bias_variable([self.lstm_output, ])
            # shape = (batch * steps, output_size)
            with tf.name_scope('Wx_plus_b'):
                self.lstm_pred = tf.matmul(l_out_x, Ws_out) + bs_out
                self.lstm_pred = tf.reshape(self.lstm_pred, [-1, self.n_steps * self.lstm_output])

        def add_nn_layer(self, inputs, in_size, out_size, activation_function=None):
            # tf.sqrt(2/in_size)是梯度爆炸/消失的处理方法
            # inputsnn_in = tf.reshape(self.lstm_pred,[-1,in_size])
            Weights = tf.Variable(tf.random_normal([in_size, out_size]) * tf.sqrt(2 / in_size))
            biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
            Wx_plus_b = tf.nn.dropout(Wx_plus_b, self.keep_prob_nn)
            if activation_function == None:
                self.pred = Wx_plus_b
            else:
                self.pred = activation_function(Wx_plus_b)

        def compute_cost(self):

            with tf.name_scope('average_cost'):
                self.cost = tf.reduce_mean(tf.square(tf.reshape(self.pred, [-1]) - tf.reshape(self.ys, [-1])))

        def _weight_variable(self, shape, name='weights'):
            initializer = tf.random_normal_initializer(mean=0., stddev=1., )
            return tf.get_variable(shape=shape, initializer=initializer, name=name)

        def _bias_variable(self, shape, name='biases'):
            initializer = tf.constant_initializer(0.1)
            return tf.get_variable(name=name, shape=shape, initializer=initializer)


    def train_func(config,reporter,dataset=xdata):
        TIME_STEPS = config["TIME_STEPS"]
        BATCH_SIZE = config["BATCH_SIZE"]
        OUTPUT_SIZE = 1
        CELL_SIZE = config["CELL_SIZE"]
        LR = config["LR"]
        k1=0
        k2=0
        N_ITER = config["N_ITER"]
        KEEP_PROB_LSTM = config["KEEP_PROB_LSTM"]
        KEEP_PROB_NN = config["KEEP_PROB_NN"]
        NN_LAYER = config["NN_LAYER"]
        LSTM_LAYER = config["LSTM_LAYER"]
        LSTM_OUTPUT = config["LSTM_OUTPUT"]
        NN_CELLSIZE = config["NN_CELLSIZE"]


        data = dataset
        data = np.array(data)
        n1 = len(data[0])-1
        train_end_index = math.floor(len(data)*0.9)
        data_train = data[0:train_end_index]
        data_test = data[train_end_index+1:]

        mean_test = np.mean(data_test,axis=0)
        std_test = np.std(data_test,axis=0)

        INPUT_SIZE = n1+1

        train_data = []
        train_target = []

        data_train = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)
        data_test = (data_test - np.mean(data_test, axis=0)) / np.std(data_test, axis=0)

        for i in range(len(data_train) - TIME_STEPS):
            x = data_train[i:i + TIME_STEPS, :]
            y = data_train[i + TIME_STEPS, n1]
            train_data.append(x.tolist())
            train_target.append(y)

        test_data = []
        test_target = []

        for i in range(len(data_test) - TIME_STEPS):
            x = data_test[i:i + TIME_STEPS, :]
            y = data_test[TIME_STEPS+i, n1]
            test_data.append(x.tolist())
            test_target.append(y)

        def get_batch():
            seq1 = np.array(train_data)
            seq1 = seq1[k1:k1+BATCH_SIZE,:,:]
            res1 = np.array(train_target)
            res1 = res1[k1:k1+BATCH_SIZE].reshape(-1,1)
            return [seq1, res1]

        def get_test_data():
            datat = np.array(test_data)
            datat = datat[k2:k2+BATCH_SIZE,:,:]
            res1 = np.array(test_target)
            res1 = res1[k2:k2 + BATCH_SIZE].reshape(-1, 1)
            return [datat, res1]

        model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LSTM_LAYER ,NN_LAYER,LR, NN_CELLSIZE, LSTM_OUTPUT)
        sess = tf.Session()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=1)
        sess.run(init)

        max_acc = 0

        acc_value = []
        for iteration in range(N_ITER):
            k1=0
            test_predict = []
            test_real = []
            for i in range(len(train_data)//BATCH_SIZE):
                seq, res = get_batch()
                k1 = k1 + BATCH_SIZE
                if iteration == 0:
                    feed_dict = {
                            model.xs: seq,
                            model.ys: res,
                            model.keep_prob_lstm:KEEP_PROB_LSTM,
                            model.keep_prob_nn:KEEP_PROB_NN
                            # create initial state
                    }
                else:
                    feed_dict = {
                        model.xs: seq,
                        model.ys: res,
                        model.keep_prob_lstm:KEEP_PROB_LSTM,
                        model.keep_prob_nn:KEEP_PROB_NN,
                        model.cell_init_state: state    # use last state as the initial state for this run
                    }

                _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict)

            if iteration%20==0:
                print('cost out: ', cost)

            k2=0
            for step in range(len(test_data) // BATCH_SIZE):
                test_x, test_y = get_test_data()
                k2 = k2 + BATCH_SIZE
                feed_dict = {
                    model.xs: test_x,
                    model.ys: test_y,
                    model.keep_prob_lstm: 1,
                    model.keep_prob_nn: 1
                    # create initial state
                 }

                pred = sess.run(
                    [model.pred],
                    feed_dict=feed_dict)
                pred = np.array(pred)
                pred = pred.reshape((-1))
                test_predict.extend(pred)
                test_y = np.array(test_y)
                test_y = test_y.reshape((-1))
                test_real.extend(test_y)
            test_predict = np.array(test_predict).reshape(-1,1)
            test_real = np.array(test_real).reshape(-1,1)
            test_predict = test_predict * std_test[n1] + mean_test[n1]
            test_real = test_real * std_test[n1] + mean_test[n1]
            acc = np.average(np.abs(test_predict - test_real[:len(test_predict)]) / test_real[:len(test_predict)])
            acc_true = 1 - acc
            print(acc_true)
            acc_value.append(acc_value)
            reporter(mean_acc = acc_true)
            if acc_true > max_acc:
                max_acc = acc_true
                if para.config.acc == []:
                    para.config.acc.clear()
                para.config.acc.append(max_acc)
        print(acc_value)
        print("**********************")



    all_trials = tune.run(
        train_func,
        name="quick start",
        stop={"mean_acc":1},
        config={"BATCH_SIZE": tune.grid_search(para.config.batch_size),
                "TIME_STEPS": tune.grid_search(para.config.time_steps),
                "CELL_SIZE": tune.grid_search(para.config.cell_size),
                "LR":tune.grid_search(para.config.lr),
                "N_ITER":tune.grid_search(para.config.n_iter),
                "KEEP_PROB_LSTM":tune.grid_search(para.config.keep_prob_lstm),
                "KEEP_PROB_NN":tune.grid_search(para.config.keep_prob_nn),
                "NN_LAYER":tune.grid_search(para.config.nn_layer),
                "LSTM_LAYER":tune.grid_search(para.config.lstm_layer),
                "LSTM_OUTPUT":tune.grid_search(para.config.lstm_output),
                "NN_CELLSIZE":tune.grid_search((para.config.nn_cellsize))
    }
    )


    best_config = all_trials.get_best_config(metric="mean_acc")
    all_config = all_trials.get_all_configs()
    print(best_config)
    print(para.config.acc)
    print(all_config)
    print("****")
    ray.shutdown()
    return [best_config,all_config]
