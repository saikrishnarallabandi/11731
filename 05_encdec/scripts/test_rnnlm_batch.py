from seq2seq_v1 import Attention as AED 
from seq2seq_v1 import EncoderDecoder as ed
from seq2seq_v1 import nnlm as LM
from seq2seq_v1 import RNNLanguageModel_batch as RNNLM_B 
import dynet as dy
import time
import math
start = time.time()
import random
from dynet import *
from utils import CorpusReader as CR

filename = '../data/en-de/train.en-de.low.en'
#filename = '../../../../dynet-base/dynet/examples/python/written.txt'
#filename = 'txt.done.data'



cr = CR(filename)
wids = cr.read_corpus_word(0)
i2w = {i:w for w,i in wids.iteritems()}

model = Model()     
trainer = SimpleSGDTrainer(model)
num_layers = 1
input_dim = 128
embedding_dim = 128
vocab_size = len(wids)
minibatch_size = 16
M = model.add_lookup_parameters((len(wids), embedding_dim))
builder = LSTMBuilder
rnnlm_b =  RNNLM_B(model, num_layers, input_dim, embedding_dim, vocab_size, M, builder)

def get_indexed(arr):
  ret_arr = []
  for a in arr:
    #print a, wids[a], M[wids[a]].value()
    
    ret_arr.append(wids[a])
  return ret_arr  

def get_indexed_batch(sentence_array):
  ret_sent_arr = []
  words_mb = 0
  for sent in sentence_array:
    ar = get_indexed(sent.split())
    ret_sent_arr.append(ar)
    words_mb += len(ar)
  return ret_sent_arr, words_mb  



# Accumulate training data
# I am using this simple version as I dont need to do tokenization for this assignment. Infact, tokenization might be bad in this case.
sentences  = []
f = open(filename)
for  line in f:
   line = line.strip()
   sentences.append( '<s>' + ' ' + line + ' ' + '</s>')

# Batch the training data ##############################################
# Sort

sentences.sort(key=lambda x: -len(x))
train_order = [x*minibatch_size for x in range(int((len(sentences)-1)/minibatch_size + 1))]



print ("startup time: %r" % (time.time() - start))
# Perform training

i = words = sents = loss = cumloss = dloss = 0
for epoch in range(100):
 random.shuffle(train_order) 
 loss = 0
 c = 1
 for sentence_id in train_order:
  #print "Processing ", sentence
  #sentence = train_order[sentence_id]
  #sentence = sentence.split() 
  #if len(sentence) > 2:  
    #print "This is a valid sentence"
  if 3 > 2:  
    #print "This is a valid sentence"
    c = c+1
    print c, " out of ", len(train_order)
    if c%250 == 1:
    #     #print "I will print trainer status now"
         trainer.status()
         print "Loss: ", loss / words
         print "Perplexity: ", math.exp(loss / words)
         print ("time: %r" % (time.time() - start))
    #     #print dloss / words
    #     loss = 0
    #     words = 0
    #     dloss = 0
    #     for _ in range(1):
   # 	     print ' '.join(k for k in sentence)
    #         samp = red.sample(nchars= len(sentence),stop=wids["</s>"])
    #         res = red.generate(get_indexed(sentence))
    #         print(" ".join([i2w[c] for c in res]).strip())
    
    #words += len(sentence) - 1
    isents, words_minibatch_indexing = get_indexed_batch(sentences[sentence_id:sentence_id+minibatch_size])
    
    #print isent
    #print "I will try to calculate error now"
    error, words_minibatch_loss = rnnlm_b.get_loss_batch(isents)
    ####### I need to fix this sometime
    #print words_minibatch_indexing , words_minibatch_loss
    #assert words_minibatch_indexing == words_minibatch_loss
    words += words_minibatch_indexing
    #print "Obtained loss ", error.value()
    loss += error.value()
    #print "Added error"
    #print error.value()
    error.backward()
    trainer.update(1.0)
 print '\n'   
 print("ITER",epoch,loss)
 print '\n'
 trainer.status()
 trainer.update_epoch(1)
    
    
