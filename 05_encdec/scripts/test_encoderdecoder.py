from seq2seq_v1 import EncoderDecoder as ed
from seq2seq_v1 import nnlm as LM
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
print "wids lengtj is", len(wids)
ED = ed(len(wids))
M = model.add_lookup_parameters((len(wids), 2))

sentences  = []
f = open(filename)
for  line in f:
   line = line.strip()
   sentences.append(line)
   
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
     loss.backward()
     #print "Back propagated the loss"
     trainer.update(1)

 print closs / sent   
 trainer.status()
 trainer.update_epoch(1.0)
 
''' 
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
