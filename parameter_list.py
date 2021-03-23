class para_list():
    def __init__(self):
        self.batch_size = [50]
        self.time_steps = [20]
        self.cell_size = [10]
        self.lr = [0.006]
        self.n_iter = [10]
        self.keep_prob_nn = [0.8]
        self.keep_prob_lstm = [0.8]
        self.nn_layer = [2]
        self.lstm_layer = [2]
        self.lstm_output = [1]
        self.nn_cellsize = [30]
        self.acc = []

config = para_list()
