from seq2seq_v1 import EncoderDecoder as ed
from seq2seq_v1 import nnlm as LM
import dynet as dy
from collections import defaultdict
from itertools import count

#w2i = defaultdict(lambda: len(wids))
w2i = defaultdict(count(0).next)
model = dy.Model()     
trainer = dy.SimpleSGDTrainer(model)
filename = '../../../../dynet-base/dynet/examples/python/written.txt'
#filename = 'txt.done.data'
lm = LM()

#w2i = defaultdict(count(0).next)
#w2i = defaultdict(lambda: len(wids))

line_array = []
f = open(filename)
for line in f:
    line = line.split('\n')[0]
    line = line.split(' ')
    for i in range(0,len(line)):
      line_array.append(line[i])

EOS = "<EOS>"
line_array.append(EOS)
#print "line_array", line_array
#w2i = defaultdict(lambda: len(line_array))
words = list([i for i in line_array])
#print "int to words", i2w
s = [w2i[word] for word in words]
i2w = list(words)
print "printign dixt", "w2i--------------------->",len(w2i), "i2w============================>",len(i2w)
wids = w2i

ED = ed(len(wids))
M = model.add_lookup_parameters((len(wids), 2))

sentences  = []
f = open(filename)
for  line in f:
   line = line.strip()
   sentences.append(line)
cum_loss = 0
words = 1
for epoch in range(10):
 sent = 0 
 for sentence in sentences:
  sent = sent  + 1 
  if len(sentence.split()) > 2:
     losses, num_words, pred = ED.encode(sentence, wids)
 #    print "wds are", num_words
#     if sent % 100 == 1:
#        print  "       ", sentence, losses.value(), cum_loss/ words 
     gen_losses = losses.vec_value()
     #print "Gen_losses: ", gen_losses
     loss = dy.sum_batches(losses)
     cum_loss += loss.value()
     words += num_words
     loss.backward()
     #print "Back propagated the loss"
     trainer.update(1)
     print "printing o/p wordsids", pred
   #  prob = []
   #  for i in range(0,len(pred.value())):
   #      if pred.value()[i] > 0:
     #        prob[i] = pred.value()[i]
     #    else:
             
   #          print pred.value()[i],i

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
