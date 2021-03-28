#!/usr/bin/env python
import os, sys, pickle
import utils
import imtpp
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def_opts = imtpp.def_opts

folder = sys.argv[1] + "/"
event_train_file = folder+"event_train.txt"
event_test_file = folder+"event_test.txt"
time_train_file = folder+"time_train.txt"
time_test_file = folder+"time_test.txt"
data = utils.read_data(
    event_train_file=event_train_file,
    event_test_file=event_test_file,
    time_train_file=time_train_file,
    time_test_file=time_test_file
)
scale = 0.1
data['train_time_out_seq'] /= scale
data['train_time_in_seq'] /= scale
data['train_time_miss_seq'] /= scale
data['test_time_out_seq'] /= scale
data['test_time_in_seq'] /= scale
data['test_time_miss_seq'] /= scale

tf.reset_default_graph()
sess = tf.Session()

imtpp_mdl = imtpp.IMTPP(
    sess=sess,
    num_categories=data['num_categories'],
    batch_size=512,
    bptt=20,
    learning_rate=0.001,
    cpu_only=False,
    _opts=imtpp.def_opts
)
imtpp_mdl.initialize(finalize=False)
imtpp_mdl.train(training_data=data)

print('Results on Test data:')
test_time_preds, test_event_preds = imtpp_mdl.predict_test(data=data)
imtpp_mdl.eval(test_time_preds, data['test_time_out_seq'], test_event_preds, data['test_event_out_seq'])