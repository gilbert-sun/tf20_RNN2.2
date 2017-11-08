# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.

Run this script on tensorflow r0.10. Errors appear when using lower versions.
"""
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 45
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006
data_label = []
data_features = []
def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    #               0               0        20*50                              50          20


    # --------------------------------------------------------------------------
    files = glob.glob("./*.csv")
    index1 = 0
    for fileNameInput in files:

        fp = open(fileNameInput, "r").readlines()
        # print("\n\n--------------------------------------Begin" + str(index1))
        # print(fileNameInput)
        # print("--------------------------------------")
        index1+=1
        cfp = csv.DictReader(fp)
        dataset1 = list(cfp)
        N = len(dataset1)
        M = int(N / 20)
        L = 20 * (M - 1) + 3
        # print("N:" + str(N) + "  M(N/20):" + str(M) + "  L[ 20 * (M-1) + 3]:" + str(L))
        # print("--------------------------------------\n\n")
        dataset2 = dataset1[:L]
        data_label = []
        data_features = []
        # print ("--------------------------------------")
        # print (M)
        # print ("--------------------------------------")
        for i in range(L):
            # print(str(i) + ":" + "\n")
            try:
                w = float(dataset2[i]["Voltage"].strip('"'))
                w1 = float(dataset2[i - 1]["Voltage"].strip('"'))
                w3 = abs(w1 - w)
                if (w3 > 0.4):
                    # print("Level4")
                    data_label.append(4)
                    data_features.append(w3)
                elif (w3 < 0.4 and w3 > 0.3):
                    # print("Level3")
                    data_label.append(3)
                    data_features.append(w3)
                elif (w3 < 0.3 and w3 > 0.2):
                    # print("Level2")
                    data_label.append(2)
                    data_features.append(w3)
                elif (w3 < 0.2):
                    # print("Level1")
                    data_label.append(1)
                    data_features.append(w3)
                    # if (i < 20):
                    #     print(data_features )
                    #     print(data_label )
            except ValueError:
                #print("Error with row", i, ":", dataset2[i])
                pass

        data_reshape = np.reshape(data_features, (M - 1, 20, 1))

        label_reshape = np.reshape(data_label,(M-1,20,1))

        #print (dataset1)
        #print("--------------------------------------")
        #print("--------------------------------------End")
    # ---------------------------------------------------------------------------


    xs = data_features[:L]

    return [data_reshape , label_reshape , xs]


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
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
        lstm_cell = rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out
            tf.summary.histogram("add_output_layer", self.pred)

    def compute_cost(self):
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(self.ms_error(
                   tf.reshape(self.pred, [-1], name='reshape_pred'),
                   tf.reshape(self.ys,   [-1], name='reshape_target')), name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    def ms_error(self, y_pre, y_target):
        return tf.square(tf.subtract(y_pre, y_target))

#-------------------------------------------------------------------------------------------gilbert adding
    def lstm_accuracy(self):
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.ys, 1)) 
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)        
#-------------------------------------------------------------------------------------------gilbert adding

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        #tf.summary.histogram("weights", initializer)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        #tf.summary.histogram("weights", initializer)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    # tf.initialize_all_variables() no long valid
    sess.run(tf.global_variables_initializer())
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    plt.ion()
    plt.show()
    index11 =0
    for i in range(2000):

        #print("\n\n---------------------------------------------------- Loop:" + str(index11))
        seq, res, xs = get_batch()
        index11 +=1

        # print (seq)
        # print (res)
        #print (xs)
        if i == 0:
            feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_init_state: state    # use last state as the initial state for this run
            }

        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)
  

        # plotting

	# print('xs:')
	# print(xs[0,:])
	# print('pred:')
	# print(pred.flatten()[:TIME_STEPS])

	# only the first 20 samples are ploted
#        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')

	# entirely plot the all 50x20 samples
#        plt.plot(xs[:].flatten(), res[:].flatten(), 'r', xs[:].flatten(), pred[:].flatten(), 'b--')

        # plt.ylim((-1.2, 1.2))
        # plt.draw()
        # plt.pause(0.3)

        if i % 100 == 0:
            print('cost: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
            print "accuracy %.5f'" % accuracy.eval(feed_dict=feed_dict)