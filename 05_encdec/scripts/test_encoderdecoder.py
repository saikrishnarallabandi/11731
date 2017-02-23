from seq2seq_v1 import EncoderDecoder as ed
from seq2seq_v1 import nnlm as LM
import dynet as dy
model = dy.Model()     
trainer = dy.SimpleSGDTrainer(model)
#filename = '../data/en-de/test.en-de.low.en'
filename = '../../../../dynet-base/dynet/examples/python/written.txt'

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
 sent = 0 
 for sentence in sentences:
  sent = sent  + 1 
  if len(sentence.split()) > 2:
     losses, num_words, pred = ED.encode(sentence, wids)
    


#     if sent % 100 == 1:
#        print  "       ", sentence, losses.value(), cum_loss/ words 
     gen_losses = losses.vec_value()
     print "Gen_losses: ", gen_losses
     loss = dy.sum_batches(losses)
     cum_loss += loss.value()
     words += num_words
     loss.backward()
     #print "Back propagated the loss"
     trainer.update(0.1)

 
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
