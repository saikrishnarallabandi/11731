from seq2seq_v1 import Attention_Batch as AED_B
from seq2seq_v1 import EncoderDecoder as ed
from seq2seq_v1 import nnlm as LM
from seq2seq_v1 import RNNLanguageModel_batch as RNNLM_B 
import nltk
import _gdynet as dy
import time
import math
start = time.time()
import random
from _gdynet import *
from utils import CorpusReader as CR

src_filename = '../data/en-de/train.en-de.low.de'
tgt_filename = '../data/en-de/train.en-de.low.en'
#filename = '../../../../dynet-base/dynet/examples/python/written.txt'
#src_filename = tgt_filename = 'txt.done.data'




cr = CR()
src_wids = cr.read_corpus_word(src_filename, 0)
tgt_wids = cr.read_corpus_word(tgt_filename, 0)
src_i2w = {i:w for w,i in src_wids.iteritems()}
tgt_i2w = {i:w for w,i in tgt_wids.iteritems()}

model = Model()     
trainer = SimpleSGDTrainer(model)
num_layers = 2
input_dim = 512
embedding_dim = 256
src_vocab_size = len(src_wids)
tgt_vocab_size = len(tgt_wids)
minibatch_size = 1

src_lookup = model.add_lookup_parameters((len(src_wids), embedding_dim))
tgt_lookup = model.add_lookup_parameters((len(tgt_wids), embedding_dim))
builder = LSTMBuilder
minibatch_size = 32
aed_b =  AED_B(len(src_wids), len(tgt_wids),  model, input_dim, embedding_dim, src_lookup, tgt_lookup, minibatch_size, builder)

def get_indexed(arr, src_flag):
  ret_arr = []
  for a in arr:
    #print a, wids[a], M[wids[a]].value()
    if src_flag == 1:
      ret_arr.append(src_wids[a])
    else:
      ret_arr.append(tgt_wids[a])
  return ret_arr  

def get_indexed_batch(sentence_array):
  ret_ssent_arr = []
  ret_tsent_arr  = []
  words_mb = 0
  for ssent,tsent in sentence_array:
    #print sent
    ar_s = get_indexed(ssent.split(),1)
    ret_ssent_arr.append(ar_s)
    ar = get_indexed(tsent.split(),0)
    ret_tsent_arr.append(ar)
    words_mb += len(ar_s)
  return ret_ssent_arr, ret_tsent_arr, words_mb  



# Accumulate training data
# I am using this simple version as I dont need to do tokenization for this assignment. Infact, tokenization might be bad in this case.
src_sentences  = []
f = open(src_filename)
for  line in f:
   line = line.strip()
   src_sentences.append( '<s>' + ' ' + line + ' ' + '</s>')

tgt_sentences  = []
f = open(tgt_filename)
for  line in f:
   line = line.strip()
   tgt_sentences.append( '<s>' + ' ' + line + ' ' + '</s>')

# Batch the training data ##############################################
# Sort
sentences = zip(src_sentences, tgt_sentences)
sentences.sort(key=lambda x: -len(x))
#train_order = [x*minibatch_size for x in range(int((len(sentences)-1)/minibatch_size + 1))]



print ("startup time: %r" % (time.time() - start))
# Perform training

i = words = sents = loss = cumloss = dloss = 0
for epoch in range(100):
 random.shuffle(sentences) 
 test = sentences[-100:]
 train = sentences[0:-100]
 loss = 0
 c = 1
 for sentence in train:
  #print "Processing ", sentence
  #sentence = train_order[sentence_id]
  #sentence = sentence.split() 
  #if len(sentence) > 2:  
    #print "This is a valid sentence"
  if 3 > 2:  
    #print "This is a valid sentence"
    c = c+1
    #print c, " out of ", len(train)
    #print ("time: %r" % (time.time() - start))
    if c%250 == 1:
    #     #print "I will print trainer status now"
         #trainer.status()
         print c, " out of ", len(train)
         print "Loss: ", loss / words
         print "Perplexity: ", math.exp(loss / words)
         print ("time: %r" % (time.time() - start))
	 test_sentence, test_reference = test[random.randint(0,9)]
         #isents, idurs, words_minibatch_indexing = get_indexed_batch(sentences[sentence_id:sentence_id+minibatch_size])
         #src,tgt = sentences[sentence_id]
         isrc = get_indexed(test_sentence.split(),1)
         ref = get_indexed(test_reference.split(),0)
         resynth = aed_b.generate(isrc)
         tgt_resynth = " ".join([tgt_i2w[cc] for cc in resynth]).strip()
         BLEUscore = nltk.translate.bleu_score.sentence_bleu([src], tgt_resynth)
         print "BLEU: ", BLEUscore
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
    #isents, idurs, words_minibatch_indexing = get_indexed_batch(sentences[sentence_id:sentence_id+minibatch_size])
    src,tgt = sentence
    isent = get_indexed(src.split(),1)
    itgt = get_indexed(tgt.split(),0)
    #print isent
    #print "I will try to calculate error now"
    error, words_minibatch_loss = aed_b.get_loss(isent,itgt)
    ####### I need to fix this sometime
    #print words_minibatch_indexing , words_minibatch_loss
    #assert words_minibatch_indexing == words_minibatch_loss
    words += words_minibatch_loss
    #print "Obtained loss ", error.value()
    loss += error.value()
    #print "Added error"
    #print error.value()
    error.backward()
    trainer.update(1.0)
 print '\n'   
 print("ITER",epoch,loss)
 print ("time: %r" % (time.time() - start))
 print '\n'
 
 trainer.status()
 trainer.update_epoch(1)
    
    
