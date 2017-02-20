from seq2seq import nnlm as LM
import numpy as np
import sys, math
import random
import math
import dynet as dy
from collections import defaultdict


lm = LM()
train_dict, wids = lm.read_corpus('../data/en-de/train.en-de.low.en')
#train_dict,wids = lm.read_corpus('txt.done.data')
#print wids
data = train_dict.items()
#print data
# Define the hyperparameters
N = 3
EVAL_EVERY = 10
EMB_SIZE = 128
HID_SIZE = 256

# Create the neural network model including lookup parameters, etc
model = dy.Model()
M = model.add_lookup_parameters((len(wids), EMB_SIZE))
W_mh = model.add_parameters((HID_SIZE, EMB_SIZE * (N-1)))
b_hh = model.add_parameters((HID_SIZE))
W_hs = model.add_parameters((len(wids), HID_SIZE))
b_s = model.add_parameters((len(wids)))
trainer = dy.SimpleSGDTrainer(model)

def calc_function(x):
    
    w_xh = dy.parameter(W_mh)
    b_h = dy.parameter(b_hh)
    W_hy = dy.parameter(W_hs)
    b_y = dy.parameter(b_s)
    #x_val = dy.inputVector(x.value())
    h_val = dy.tanh(w_xh * x + b_h)
    y_val = W_hy * h_val + b_y
    return dy.softmax(y_val)
  
def do_loss(probs, label):
 label = wids[label]
 #print "Label: ", label
 k = dy.pick(probs,label)
 print "k: ",k.value()
 return k
 #return -math.log(t)
  
# Training
for epoch in range(10000):
  epoch_loss = 0
  random.shuffle(data)
  c = 1
  for x , ystar in data:
    c = c + 1
    dy.renew_cg()
    z = M[wids[x.split()[0]]].value() + M[wids[x.split()[1]]].value()
    y = calc_function(dy.inputVector(z))
    err = dy.pickneglogsoftmax(y, wids[ystar])
    epoch_loss = epoch_loss + err.value()
    err.backward()
    trainer.update()
    if c % 5000 == 1:
      print "    ", x, err.value(), ystar
  print("Epoch %d: loss=%f" % (epoch, epoch_loss))  
    


