import os
import time
import tensorflow as tf
import _pickle as pickle
import numpy as np

from model import Model
from w_initialization import xavier_weight_init
from utils.general_utils import Progbar
from DataProcessing import getMeanDay, getMeanDaybyUser

from sklearn.utils import shuffle


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_items = 36
    n_users = 38
    n_mean = 0
    n_f = 5
    lamb2 = 25
    lamb3 = 10
    lamb4 = 0.02 #0.02
    lamb5 = 0.01 #10
    lamb6 = 0.5 #15
    beta = 0.4
    n_epochs = 30
    lr = 0.002 #0.005
    item_bin_size = 60
    batch_size = 2048
    weight_filename = "data\\weights\\recommanding_with_user_time_drifting.weight"
    
class RecommandationModel(Model):
    """
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        """
        self.user_placeholder = tf.placeholder(tf.int32, [None, ], name="user")
        self.item_placeholder = tf.placeholder(tf.int32, [None, ], name="item")
        self.rank_placeholder = tf.placeholder(tf.float32, [None, ], name="rank")
        self.tbin_placeholder = tf.placeholder(tf.int32, [None, ], name="tbin")
        self.tday_placeholder = tf.placeholder(tf.int32, [None, ], name="tday")
        self.mean_ud_placeholder = tf.placeholder(tf.float32, [None,], name = "mean_day_by_user")
        
        self.global_mean_placeholder = tf.placeholder(tf.float32, shape=(), name = "global_mean_rank")


    def create_feed_dict(self, input_df, mean_rank, mean_u_day): #label_indices, label_data, label_shape, input_rank_data, input_tb_data, input_td_data, mean_rank, mean_u_day):
        feed_dict = {}

        feed_dict[self.user_placeholder] = input_df["userID"].values.astype(int)
        feed_dict[self.item_placeholder] = input_df["itemID"].values.astype(int)
        feed_dict[self.rank_placeholder] = input_df["overall"].values
        feed_dict[self.tbin_placeholder] = input_df["ITBin"].values.astype(int)
        feed_dict[self.tday_placeholder] = input_df["ReviewDay"].values.astype(int)
        feed_dict[self.global_mean_placeholder] = mean_rank
        feed_dict[self.mean_ud_placeholder] = mean_u_day.astype(float)
             
        return feed_dict

    def add_prediction_op(self):

        weight_initializer = xavier_weight_init()
        self.WPI = tf.get_variable("item_vector", shape = [self.config.n_items, self.config.n_f], 
                               initializer = weight_initializer)  #I*F
        self.WPU = tf.get_variable("user_vector", shape = [self.config.n_users, self.config.n_f], 
                               initializer =  weight_initializer)  #U*F
        self.BU = tf.get_variable("bias_user", shape= [self.config.n_users,], initializer = tf.zeros_initializer())  #U
        self.BI = tf.get_variable("bias_item", shape = [self.config.n_items,], initializer =tf.zeros_initializer()) 

        """
        self.WBIT = tf.get_variable("bias_item_bin", shape = [self.config.n_items, self.config.item_bin_size],
                                    initializer = tf.zeros_initializer())
        self.Alpha = tf.get_variable("dev_weight", shape = [self.config.n_users,], initializer = weight_initializer) 
        """        
        #get the time bin values for each row
        #self.bias_item_binvalue = tf.gather_nd(self.WBIT, tf.stack([self.item_placeholder, self.tbin_placeholder], axis=1), name = "item_time_bin_value") 
        
        bias_user = tf.nn.embedding_lookup(self.BU, self.user_placeholder, name = "bias_user") 
        bias_item = tf.nn.embedding_lookup(self.BI, self.item_placeholder, name = "bias_item")        
        """
        mean_tday = tf.nn.embedding_lookup(self.mean_ud_placeholder, self.user_placeholder) #mean tday by user t(mean)
        tday_diff = tf.subtract(tf.to_float(self.tday_placeholder), mean_tday) # t - t(mean)
        alpha_value = tf.nn.embedding_lookup(self.Alpha, self.user_placeholder)
        dev_t = tf.multiply(tf.sign(tday_diff), tf.pow(tf.abs(tday_diff), self.config.beta)) #sign(t - t(mean))*abolute(t - t(mean))**beta
        """
        user_vector = tf.nn.embedding_lookup(self.WPU, self.user_placeholder)
        item_vector = tf.nn.embedding_lookup(self.WPI, self.item_placeholder)
        """
        self.bias_user_tvalue =  tf.multiply(alpha_value, dev_t)
        bias_user_time = tf.add(bias_user, self.bias_user_tvalue)
        bias_item_time = tf.add(bias_item, self.bias_item_binvalue)
        """
        bias_vector = tf.reduce_sum(tf.multiply(user_vector, item_vector), 1)
        
        pred = tf.add(self.global_mean_placeholder, bias_user) #_time)
        pred = tf.add(pred, bias_item) #_time)
        pred = tf.add(pred, bias_vector)
        
        self.test_pred = tf.add(self.global_mean_placeholder, bias_user)
        self.test_pred = tf.add(self.test_pred, bias_item)
        self.test_pred = tf.add(self.test_pred, bias_vector)
        
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (users, items) containing the prediction of ranks
        Returns:
            loss: A 0-d tensor (scalar)
        """        
        loss =  tf.nn.l2_loss(tf.subtract(pred ,self.rank_placeholder)) + 0.5*self.config.lamb4*(tf.nn.l2_loss(self.BU) + tf.nn.l2_loss(self.BI)+tf.nn.l2_loss(self.WPU) + tf.nn.l2_loss(self.WPI)) 
        #+ 0.5*self.config.lamb5*tf.nn.l2_loss(self.bias_item_binvalue) + 0.5*self.config.lamb6*tf.nn.l2_loss(self.bias_user_tvalue)

        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from l2_loss.
        Returns:
            train_op: The Op for training.
        """

        train_op = tf.train.AdamOptimizer(learning_rate = self.config.lr).minimize(loss)

        return train_op

    def train_on_batch(self, sess, input_df, mean_rank, mean_u_day): #, dev_set):

        feed = self.create_feed_dict(input_df, mean_rank, mean_u_day)
        _, pred = sess.run([self.train_op, self.pred], feed_dict=feed)
        return pred

    def run_epoch(self, sess, train_df, mean_rank, mean_u_day, dev_df): 
        
        shuffled_df = shuffle(train_df)
        
        num_loop = (len(shuffled_df.index) + 1) // self.config.batch_size if (len(shuffled_df.index) + 1) % self.config.batch_size == 0 else (len(shuffled_df.index) + 1 ) // self.config.batch_size + 1

        for i in range(num_loop):
            self.train_on_batch(sess, shuffled_df[i*self.config.batch_size: (i+1)*self.config.batch_size], mean_rank, mean_u_day)
        
        print ("Evaluating on dev set")
                                                     
        ###Evaluate on DEV SET                                                                     
        dev_pred = sess.run(self.test_pred, feed_dict=self.create_feed_dict(dev_df, mean_rank, mean_u_day))                      
        dev_loss = sum((dev_pred - dev_df["overall"])**2) / len(dev_pred)                    
        return dev_loss

    def fit(self, sess, saver, train_df, dev_df):
        best_dev_UAS = 1000

        mean_rank = train_df["overall"].mean() #global rank mean
        
        mean_u_day = getMeanDaybyUser(train_df)


        for epoch in range(self.config.n_epochs):
            print ("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_UAS = self.run_epoch(sess, train_df, mean_rank, mean_u_day, dev_df)

            print("current DEV loss = ", dev_UAS)

            if dev_UAS < best_dev_UAS:
                best_dev_UAS = dev_UAS
                print('new best dev loss!:', best_dev_UAS)
                if saver:
                    print( "New best dev UAS! " + self.config.weight_filename)
                    saver.save(sess, self.config.weight_filename)
            print()


    def __init__(self, config): 
        self.config = config
        self.build()

