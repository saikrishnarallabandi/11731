from seq2seq import nnlm as LM
import numpy as np
import sys, math
import random
import math
import dynet as dy
from collections import defaultdict


lm = LM()
#train_dict, wids = lm.read_corpus('../data/en-de/train.en-de.low.en')
train_dict,wids = lm.read_corpus('txt.done.data')
#print wids
data = train_dict.items()
#print data
# Define the hyperparameters
#N = 3
#EVAL_EVERY = 10
#EMB_SIZE = 128
#HID_SIZE = 256

model = dy.Model()
trainer = dy.SimpleSGDTrainer(model)

best_score = None
token_count = sent_count = cum_loss = cum_perplexity = 0.0
sample_num = 0
import time
_start = time.time()
print_flag = 0
print time.time()
  
# Training
for epoch in range(10):
  random.shuffle(data)
  if print_flag == 1:
    print epoch, sample_num, " " , trainer.status()
    print "L: ", cum_loss / token_count
    print "P: ", math.exp(cum_loss / token_count)
    print "T: ", ( time.time() - _start)
  _start = time.time()
  losses = lm.build_nnlm_graph(data)
  print _start
  gen_losses = losses.vec_value()
  loss = dy.sum_batches(losses)
  cum_loss += loss.value()
  cum_perplexity += sum([math.exp(gen_loss)] for gen_loss in gen_losses)
  token_count += len(data)
  sent_count += len(sents)
  
  loss.backward()
  trainer.update(learning_rate)
  sample_num += len(sents)
  trainer.update_epoch(1)
  print_flag = 1
  print "Epoch 0"
  

'''
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
    
'''

