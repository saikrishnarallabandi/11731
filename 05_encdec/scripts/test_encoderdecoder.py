from seq2seq_v1 import EncoderDecoder as ed
from seq2seq_v1 import nnlm as LM
from seq2seq_v1 import RNNLanguageModel 
import dynet as dy
import random
from dynet import *
model = Model()     
trainer = SimpleSGDTrainer(model)
#filename = '../data/en-de/test.en-de.low.en'
#filename = '../../../../dynet-base/dynet/examples/python/written.txt'
filename = 'txt.done.data'

lm = LM()
train_dict,wids = lm.read_corpus(filename)
i2w = {i:w for w,i in wids.iteritems()}
print "wids lengtj is", len(wids)
ED = ed(len(wids))
M = model.add_lookup_parameters((len(wids), 128))
rnnlm =  RNNLanguageModel(model, 2, 128, 128, len(wids), builder=LSTMBuilder)

def get_indexed(arr):
  ret_arr = []
  for a in arr:
    #print a, wids[a], M[wids[a]].value()
    
    ret_arr.append(wids[a])
  return ret_arr  

sentences  = []
f = open(filename)
for  line in f:
   line = line.strip()
   sentences.append(line)

words = sents = loss = cumloss = 0
for epoch in range(100):
 random.shuffle(sentences) 
 c = 1
 for sentence in sentences:
  sentence = sentence.split() 
  if len(sentence) > 2:  
    c = c+1
    if c%250 == 1:
         trainer.status()
         print loss / words
         loss = 0
         words = 0
         #for _ in range(1):
         #    samp = rnnlm.sample(first=wids["<s>"],stop=wids["</s>"])
         #    print("".join([i2w[c] for c in samp]).strip())
    words += len(sentence) - 1
    isent = get_indexed(sentence)
    #print isent
    error = rnnlm.build_lm_graph(isent)
    loss += error.scalar_value()
    error.backward()
    trainer.update(1.0)
 print '\n'   
 print("ITER",epoch,loss)
 print '\n'
 trainer.status()
 trainer.update_epoch(1)
    
    
         

''''   
cum_loss = 0
words = 1
for epoch in range(100):
 random.shuffle(sentences) 
 sent = 0 
 closs =  0 
 for sentence in sentences:
  sent = sent  + 1 
  if len(sentence.split()) > 2:
     loss = ED.encode(sentence, wids)
     closs += loss.scalar_value()
     loss_value = loss.value()
     print "                              " , loss_value, sentence
     loss.backward()
     #print "Back propagated the loss"
     trainer.update(1)

 print closs / sent   
 trainer.status()
 trainer.update_epoch(1.0)
 
enc = dy.LSTMBuilder(1, 2,128,model)
#print "Got an encoder"
state = enc.initial_state()
#print "Added initial state"
#state.add_input(M[wids['help']])
#print "Added help"
state = enc.initial_state(losses)
#print "Initialized it to last stage"
state.add_input(M[wids['help']])
'''
