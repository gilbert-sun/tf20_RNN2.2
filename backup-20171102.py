# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.

Run this script on tensorflow r0.10. Errors appear when using lower versions.
"""
#!/usr/bin/python -u
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import random
import os,sys
import time

BATCH_START = 50
TIME_STEPS = 20
BATCH_SIZE = 0
INPUT_SIZE = 3 #1
OUTPUT_SIZE = 4
CELL_SIZE = 36
RNN_LAYER_SIZE= 1
DROP_OUT_RATE = 0.5
IS_TRAINING =1
LR = 0.001
BASE_SIZE_DIV = 0

snap_shot = 1000


Vdata_reshape_global  = np.array([])

Sdata_reshape_global  = np.array([])

Tdata_reshape_global  = np.array([])

label_reshape_global = np.array([])

Total_Dlength_global = 0
expected_sparse_output =[]

def read_batch(isTrain):
    if(isTrain):
        return get_batch("./OutCsv/amita/*.csv" , 37)
    else:
        return get_batch("./Exa_2017091x/*.csv" , 10)


def get_batch(folderName , EPOCH_No):

    # --------------------------------------------------------------------------
    files = glob.glob(folderName)
    index1 = 0
    Total_data_lens = 0

    global Vdata_reshape_global
    global Sdata_reshape_global
    global Tdata_reshape_global
    global label_reshape_global
    global Total_Dlength_global

    Vdata_reshape_all = np.array([])

    Tdata_reshape_all = np.array([])

    Sdata_reshape_all = np.array([])

    label_reshape_all = np.array([])

    Vdata_reshape_global = Vdata_reshape_all

    Sdata_reshape_global = Sdata_reshape_all

    Tdata_reshape_global = Tdata_reshape_all

    label_reshape_global = label_reshape_all

    for fileNameInput in files:

        fp = open(fileNameInput, "r")#.readlines()

        index1 += 1
        cfp = csv.DictReader(fp)

        dataset1 = list(cfp)

        N = len(dataset1)


        M = int(N / 20)
        L = 20 * (M)   # (M - 1)+ 3
        #Total_data_lens = Total_data_lens+ L
        # print("Total N:" + str(N) + "  M(N/20):" + str(M) + "  L[ 20 * (M-1) + 3]:" + str(L))
        # print("--------------------------------------\n\n")
        dataset2 = dataset1[:L]
        dataset2 = np.concatenate((dataset2, dataset2[::-1]), axis=0)
        L = len(dataset2)

        for i in range(L):
            # print(str(i) + ":" + "\n")
            try:
                w = float(dataset2[i]["BTV"].strip('"'))
                w1 = float(dataset2[i - 1]["BTV"].strip('"'))

                S = float(dataset2[0]["SOC"].strip('"'))

                T = float(dataset2[i]["TJ1"].strip('"'))

                if(i == 0  ):
                    w3 = 0
                else:
                    w3 = abs(w1 - w)

                Vdata_reshape_all = np.concatenate((Vdata_reshape_all, [w3]), axis=0)
                Sdata_reshape_all = np.concatenate((Sdata_reshape_all,[S]),axis=0)
                Tdata_reshape_all = np.concatenate((Tdata_reshape_all, [T]), axis=0)

                if ((w3 >= 50 or  T >= 860 ) and  (S >= 75) ):# and SOC == 100
                    label_reshape_all = np.concatenate((label_reshape_all, [0,0,0,1]), axis=0)

                elif ( ((w3 < 50 and w3 >= 35)  or  ( T >= 1000)) and (  S >= 50) ): # and SOC == 75
                    label_reshape_all = np.concatenate((label_reshape_all, [0,0,1,0]), axis=0)


                elif ( ((w3 < 50 and w3 >= 35)  or  (T  < 860 and T >= 790)) and (S < 76 and S >= 60) ): # and SOC == 75
                    label_reshape_all = np.concatenate((label_reshape_all, [0,1,0,0]), axis=0)


                elif ( ((w3 < 35 and w3 >= 25)  or  (T  < 790 and T >= 600)) and (S < 60 and S >= 45) ): # and SOC == 75

                    label_reshape_all = np.concatenate((label_reshape_all, [1,0,0,0]), axis=0)

                elif ((w3 < 25  or   T  < 600 ) and (S< 45) ): # and SOC == 50
                    label_reshape_all = np.concatenate((label_reshape_all,[1,0,0,0]) ,axis=0)

                else:
                    label_reshape_all = np.concatenate((label_reshape_all,[1,0,0,0]) ,axis=0)
                    # label_reshape_all[Total_data_lens + i - 1] = 100.0

            except ValueError:
                # print("Error with row", i, ":", dataset2[i])
                pass

        Total_data_lens = Total_data_lens + L


        fp.close()

    BASE_SIZE_DIV = 1000
    PEAK_BATCH = 1000

    N1 = 120
    N0 = int(Total_data_lens / N1 / BASE_SIZE_DIV)

    RADON_SHIFT = EPOCH_No % N0
    # RADON_SHIFT = EPOCH_No % 38

    print("%%   ---------- %%%%%%%%%%%--------------N0:" + str(Total_data_lens) + ":EPOCH:" + str(EPOCH_No) + "Total_line" + str(
        N0) + "Rad_shift" + str(RADON_SHIFT))

    M1 = 20 * (N1) * RADON_SHIFT
    M2 = 20 * (N1) * (RADON_SHIFT + 1)


    if(True):
        print("\n-[in Sub-Func : go_batch ]----------------------- 0.1 " + " FileNo.:" + str(index1) + "files")

        print("[----------label]Out_Label_check_0-30:")
        print (np.shape(label_reshape_all))
        print(label_reshape_all[M1:M1+30])
        print("Volt_Label_check_0-30:")
        print(Vdata_reshape_all[M1:M1+30])
        print("Temp_Label_check_0-30:")
        print(Tdata_reshape_all[M1:M1 +30])
        print("SOC_Label_check_0-30:")
        print(Sdata_reshape_all[M1:M1 +30])
    Vdata_reshape_all =  Vdata_reshape_all[:,np.newaxis]
    Sdata_reshape_all =  Sdata_reshape_all[:,np.newaxis]
    Tdata_reshape_all =  Tdata_reshape_all[:,np.newaxis]

    Vdata_reshape_global = Vdata_reshape_all
    Sdata_reshape_global = Sdata_reshape_all
    Tdata_reshape_global = Tdata_reshape_all
    label_reshape_global = label_reshape_all
    Total_Dlength_global = N1

    if(True):
        print ("-----Random F-Index_No.:----- 0.01:" + "( " + str(RADON_SHIFT) + " )")
        print ( "M1:"+str(M1)+ " M2:"+str(M2)+ " Total_20/N:" +str(N1))
    #data_Endreshape3D = np.concatenate((Tdata_reshape_all[M1:M2],Vdata_reshape_all[M1:M2]),axis=1)

    data_Endreshape3D = np.concatenate((Sdata_reshape_all[M1:M2], Tdata_reshape_all[M1:M2] ,Vdata_reshape_all[M1:M2]),axis=1)
    xs1 = data_Endreshape3D
    if(True):
        print("---seq.shape : seq: ")
        print(np.shape(data_Endreshape3D))
        print( data_Endreshape3D [0:30] )
    data_reshape1 = np.reshape(data_Endreshape3D, ( N1, 20, 3)) #N1, 20, 3
    if(True):
        print("---seq.re-shape : seq: ")
        print(np.shape(data_reshape1))
        print( data_reshape1 [0:30] )


        print ("------------------------ 0.02")
        print (np.shape(label_reshape_all))

    #     every label is [0 0 0 1] , so that M1*4 , M2 = (M1+1)*4
    label_Endreshape = label_reshape_all[M1*4:M2*4]
    print(np.shape(label_Endreshape))
    #label_Endreshape = label_reshape_all[:20*(Total_data_lens / BASE_SIZE_DIV)]
    label_reshape1 = np.reshape(label_Endreshape, (-1, 4))
    print(np.shape(label_reshape1))
    # if (EPOCH_No % PEAK_BATCH == 0):
    print((label_reshape1[0:30]))
    print("---[shape:xs1]---[xs1]------------------ 0.03")

    if (True):
        print(np.shape(xs1))
        print((xs1[0:30]))
        print("[exit go_batch]------------------------ 0.04")

    return [data_reshape1, label_reshape1, xs1, N1 , Total_data_lens ,Vdata_reshape_global,Sdata_reshape_global ,Tdata_reshape_global ,label_reshape_global]



def get_batch1(EPOCH_No, Totol_length ,Vdata_reshape,Sdata_reshape ,Tdata_reshape ,label_reshape):


    BASE_SIZE_DIV = 1000
    PEAK_BATCH = 1000

    N1 = 120
    N0 = int (Totol_length / N1 / BASE_SIZE_DIV )


    RADON_SHIFT = EPOCH_No % N0
    # RADON_SHIFT = EPOCH_No % 38

    # print("%%%%%%%%%%%%%%%%%%%%%--------------N0:" + str(Totol_length) + ":EPOCH:" + str(EPOCH_No)+ "Total_line" + str(N0)+ "Rad_shift" + str(RADON_SHIFT))


    M1 = 20 * (N1) * RADON_SHIFT
    M2 = 20 * (N1) * (RADON_SHIFT + 1)

    if (False):
        print("--------------------------------Random F-Index_No.:----- 0.01:" + "( " + str(RADON_SHIFT) + " )")
        print("M1:" + str(M1) + " M2:" + str(M2) + " Total_20/N:" + str(N1))
    # data_Endreshape3D = np.concatenate((Tdata_reshape_all[M1:M2],Vdata_reshape_all[M1:M2]),axis=1)

    data_Endreshape3D = np.concatenate((Sdata_reshape[M1:M2], Tdata_reshape[M1:M2], Vdata_reshape[M1:M2]),
                                       axis=1)
    xs1 = data_Endreshape3D
    if (False):
        print("------------dddddd------------------seq.shape : seq: ")
        print(np.shape(data_Endreshape3D))
        print(data_Endreshape3D[0:100])
    data_reshape1 = np.reshape(data_Endreshape3D, (N1, 20, 3))  # N1, 20, 3
    if (False):
        print("------------------------------seq.re-shape : seq: ")
        print(np.shape(data_reshape1))
        print(data_reshape1[0:100])

        print("--------------------------------------------------- 0.02")
        print(np.shape(label_reshape))

    # every label is [0 0 0 1] , so that M1*4 , M2 = (M1+1)*4
    label_Endreshape = label_reshape[M1 * 4:M2 * 4]
    if (False):
        print(np.shape(label_Endreshape))
    # label_Endreshape = label_reshape_all[:20*(Total_data_lens / BASE_SIZE_DIV)]
    label_reshape1 = np.reshape(label_Endreshape, (-1, 4))

    if (False):
        print(np.shape(xs1))
        print((xs1[0:30]))
        print("---------------------------[exit go_batch]------------------------ 0.04")

    return [data_reshape1, label_reshape1, xs1, N1]

def DEBUG_MSG(DEG_Switch, NO, MSG ):
    if(DEG_Switch):
        print(str (NO)+ "--------------------------Begin")
        print (np.shape(MSG))

        print("---------------------------End!\n")


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size ,rnn_total_layer,drop_out_rate, is_training):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size

        self.batch_size = batch_size
        self.accuracy = 0
        self.rnn_layer_size = rnn_total_layer
        self.dropout_rate = drop_out_rate


        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys')

        with tf.variable_scope('in_hidden'):
            self.add_input_layer()

        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('accurary'):
            self.compute_accuracy( )

    def add_input_layer(self, ):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size) (3, 36)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, ) (36,)
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in

        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')


    def add_cell(self):
        lstm_cell = rnn.BasicLSTMCell( self.cell_size , forget_bias=1.0, state_is_tuple=True)

        lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_rate)

        lstm_cell = rnn.MultiRNNCell([lstm_cell] * self.rnn_layer_size )

        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state( self.batch_size , dtype=tf.float32)

        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)




    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')

        Ws_out = self._weight_variable([self.cell_size, self.output_size])

        bs_out = self._bias_variable([self.output_size, ])

        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.nn.softmax (tf.matmul(l_out_x, Ws_out) + bs_out)

            self.pred1 = tf.matmul(l_out_x, Ws_out) + bs_out

            tf.summary.histogram("add_output_layer", self.pred)

    def compute_cost(self):
        with tf.name_scope('average_cost'):

            self.cost = tf.div(
                tf.reduce_sum(self.ms_error(
                    tf.reshape(self.pred1, [-1,4], name='reshape_pred'),
                    tf.reshape(self.ys, [-1,4], name='reshape_target')),
                    name='losses_sum'), self.batch_size,name='average_cost')

            tf.summary.scalar('cost', self.cost)



    def compute_accuracy(self):
        with tf.name_scope('average_accuracy'):
            self.correct_pred  = tf.equal(  tf.argmax(  self.pred1,  1,name = "X" ) ,  tf.argmax( self.ys , 1 ,name = "Y"))

            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            tf.summary.scalar('accuracy', self.accuracy)


    def ms_error(self, y_pre, y_target):
        return tf.square(tf.subtract(y_pre, y_target))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        # tf.summary.histogram("weights", initializer)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        # tf.summary.histogram("weights", initializer)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)



if __name__ == '__main__':
    IS_TRAINING = False

    BegTime = '{}{}{}'.format("./result_3LSTM_20Epoch_", time.ctime(), "_.txt")

    BegTime1 = '{}{}{}'.format("./result_3LSTM_1000Epoch_", time.ctime(), "_.txt")

    seq1, res1, xs1 , full_length1,full_length2 ,Vdata_reshape1,Sdata_reshape1 ,Tdata_reshape1 ,label_reshape1  =  get_batch("./OutCsv/amita/*.csv" , 0)  #read_batch(True)#
    print(" \n\n0 -------------------------------------------------------------------")
    print(len(seq1))
    print(len(res1))
    print(len(xs1))
    print(full_length1)
    BATCH_SIZE = full_length1
    tf.reset_default_graph()
    print("1 ---------------------------------------------------------------"+ str (BATCH_SIZE) +"----\n\n")
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE , full_length1 , RNN_LAYER_SIZE , DROP_OUT_RATE, False )
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs",   graph =sess.graph)

    # test_writer = tf.summary.FileWriter("tlogs", sess.graph)

    # ----------------------------------------------------- 2017/10/10 Begin
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state("./model")
    # tf.train.checkpoint_exists(ckpt.model_checkpoint_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        sess.run(tf.global_variables_initializer())
    # ----------------------------------------------------- 2017/10/10 End


    index11 = 0
    for i in range(11):

        if (i % 1 == 0):
            print("\n\nBegin----Battery Training  Entre -------------------------------------------- Epoch:" + str(index11))
        seq, res, xs , full_length =  get_batch1(i,full_length2,Vdata_reshape1,Sdata_reshape1 ,Tdata_reshape1 ,label_reshape1)

        index11 += 1

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
                model.cell_init_state: state  # use last state as the initial state for this run
            }

        cost, state, pred, accu = sess.run(
            [model.cost, model.cell_final_state, model.pred,model.accuracy],
            feed_dict=feed_dict)

        with sess.as_default():
            print("----------------------0--------------------------")
            print(np.shape(feed_dict))

            print(res[20:120])
            print("----------------------1--------------------------")

            print (model.pred.eval(feed_dict = feed_dict)[20:120])
            print("----------------------2--------------------------")
            print(model.correct_pred.eval(feed_dict=feed_dict)[20:120])

            print("------------------------------------------------")
        # with sess.as_default():
        #     print (model.pred.eval())
        if (i % 100 == 0):
            print("\n---predict begin [0:20]-----\n")
            print (pred[0:20])
            # print("\n---label begin [0:20]----\n")
            # plotting
            print (res[0:20])

            print("\n---accu compare result [0:20]----\n")
            print(accu)
            print ( "\nEnd----Battery Training Exit----------------ddd---------------------------- Epoch:("+ str(i)+")\n")


        if ((i % 1000 == 0) or i < 150):
            print('cost: ', round(cost, 6))
            result = sess.run(merged, feed_dict)
            #print (feed_dict.values())
            writer.add_summary(result, i)
            print("\nStep " + str(i) + ", Minibatch Loss= " + \
                  "{:.10f}".format(cost) + ", Training Accuracy= " + \
                  "{:.10f}".format(accu)+  ", Out Prediction= "  + \
                  "{:.10}".format( str(pred) ) + "\n")
        if i % 20 == 0:
            fp1 = open(BegTime ,'a+')
            fp1.write("\nStep " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(cost) + ", Training Accuracy= " + \
                  "{:.6f}".format(accu)  + "\n")
            fp1.close()
        if i % 100 == 0:
            fp1 = open(BegTime1 ,'a+')
            fp1.write("\nStep " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(cost) + ", Training Accuracy= " + \
                  "{:.6f}".format(accu)  + "\n")
            fp1.close()

        if i % snap_shot == 0:
            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join("./model", "lstm.ckpt")
            saver.save(sess, checkpoint_path)
    # Save checkpoint and zero timer and loss.
    checkpoint_path = os.path.join("./model", "lstm.ckpt")
    saver.save(sess, checkpoint_path)
    sess.close()