from collections import defaultdict
from itertools import count
import sys
from _dynet import *
import _dynet as dy

class RNNLanguageModel_batch:
  
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, lookup, builder=SimpleRNNBuilder):
        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

        self.lookup = lookup
        self.R = model.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
        self.bias = model.add_parameters((VOCAB_SIZE))

    def save_to_disk(self, filename):
        model.save(filename, [self.builder, self.lookup, self.R, self.bias])

    def load_from_disk(self, filename):
        (self.builder, self.lookup, self.R, self.bias) = model.load(filename)
        
    def sample(self, first=1, nchars=0, stop=-1):
        res = [first]
        renew_cg()
        state = self.builder.initial_state()

        R = parameter(self.R)
        bias = parameter(self.bias)
        cw = first
        while True:
            x_t = lookup(self.lookup, cw)
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            ydist = softmax(r_t)
            dist = ydist.vec_value()
            rnd = random.random()
            for i,p in enumerate(dist):
                rnd -= p
                if rnd <= 0: break
            res.append(i)
            cw = i
            if cw == stop: break
            if nchars and len(res) > nchars: break
        return res

    
    def run_lstm_batch(self, sent_array):
        print "Minibatch: ", len(sent_array)
        renew_cg()
        init_state = self.builder.initial_state()
        R = parameter(self.R)
        bias = parameter(self.bias)
        wids = []
        masks = []         
        # get the wids and masks for each step
        # "I am good", "This is good", "Good Morning" -> [['I', 'Today', 'Good'], ['am', 'is', 'Morning'], ['good', 'good', '<S>'], ['I', 'Today', 'Good'], ['am', 'is', 'Morning'], ['good', 'good', '<S>']]
        tot_words = 0
        wids = []
        masks = []
        for i in range(len(sent_array[0])):
           wids.append([
                (sent[i] if len(sent)>i else 3) for sent in sent_array])
           mask = [(1 if len(sent)>i else 0) for sent in sent_array]
           masks.append(mask)
           tot_words += sum(mask)
        # start the rnn by inputting "<s>"
        init_ids = [2] * len(sent_array)
        #print dy.lookup_batch(self.lookup,init_ids)
        print "Looked up"
        s = init_state.add_input(dy.lookup_batch(self.lookup,init_ids))
   
        # feed word vectors into the RNN and predict the next word
        losses = []
        for wid, mask in zip(wids, masks):
           # calculate the softmax and loss
           print "WID ", wid , " out of ", len(wids)
           score = dy.affine_transform([bias, R, s.output()])
           print "Calculated Score"
           loss = dy.pickneglogsoftmax_batch(score, wid)
           print "Got Loss"
           # mask the loss if at least one sentence is shorter
           if mask[-1] != 1:
              mask_expr = dy.inputVector(mask)
              mask_expr = dy.reshape(mask_expr, (1,), len(sent_array))
              loss = loss * mask_expr
           losses.append(loss)
           print "Appended loss"
           # update the state of the RNN    
           wemb = dy.lookup_batch(self.lookup, wid)
           s = s.add_input(wemb) 
           print "Added embedding"
  
        return dy.sum_batches(dy.esum(losses)), tot_words  
      
      
      
      
      
