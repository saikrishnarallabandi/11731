#from seq2seq import ngramlm, neurallm
import cPickle, pickle
import numpy as np
import os
from collections import defaultdict

class loglinearlm:
     
     def __init__(self):
         self.feats_and_values ={}
         self.wids = defaultdict(lambda: len(self.wids))
         
     def read_corpus(self, file):
       # for each line in the file, split the words and turn them into IDs like this:
       print file
       #self.accumulate_trigramfeatures(file)
       #self.accumulatebigramfeatures(file)
       #self.accumulateskipgramfeatures(file)
       f = open(file)
       self.data_array_train = []
       for line in f:
          line = '<s> ' + line.split('\n')[0] + ' </s>'
          
          self.data_array_train.append(line)
          words = line.split()
          for word in words:
              wid = self.wids[word]
       f.close()
       #self.get_feature_vectors(file)  
       
     
     def accumulate_trigramfeatures(self, file):
        self.trigramfeaturedict = {}
        g = open(file)
        for line in g:
	  line = '<s> ' + line.split('\n')[0] + ' </s>'
	  line = line.split()
	  contexts = zip(line[0:len(line)-2], line[1:len(line)-1], line[2:])
	  for prev_2, prev_1, current in contexts:
	    #print prev_2, prev_1, current
	    context = prev_2 + ' ' + prev_1
	    self.trigramfeaturedict[context] = current
	g.close()    

     def print_words(self):
        for wid in self.wids:
            print wid, self.wids[wid]
     
     def get_vocab_size(self):
          return len(self.wids)
     
     def calculate_feature_F1(self, file):
        # This is a trigram context feature
        features = []
        feature_vector_prime = np.zeros(self.get_vocab_size())
        g = open(file)
        for line in g:
	  line = '<s> ' + line.split('\n')[0] + ' </s>'
	  #print line
	  line = line.split()
	  contexts = zip(line[0:len(line)-2], line[1:len(line)-1], line[2:])
	  for prev_2, prev_1, current in contexts:
	    feature_vector = feature_vector_prime
	    #print prev_2, prev_1, current, self.get_vocab_size()
	    #print prev_2, self.wids[prev_2], feature_vector
	    prev_2_id = self.wids[prev_2]
	    feature_vector[prev_2_id] = 1.0
	    prev_1_id = self.wids[prev_1]
	    feature_vector[prev_1_id] = 1.0
	    features.append(feature_vector)
            #print feature_vector
        #print features[0]   
        g.close()
	return features    
	           
        
     def sparse_features_to_dense_features(self, features):
        ret = np.zeros(len(features))
        print ret
        for f in features:
             print f
             ret[f] += 1
        return ret

     def get_feature_vectors(self, file):
            features = []
            features.append(self.sparse_features_to_dense_features(self.calculate_feature_F1(file)))
            #features.append(calculate_feature_F2())
            #features.append(calculate_feature_F3())
            #features.append(calculate_feature_F4())
            return features 
         
  
  
class ngramlm:
  
     def __init__(self, order):
             self.order = order
             self.unigrams = {}
             self.bigrams = {}
             self.alpha_unk = 0.02
             self.alpha_1 = 0.245
             self.alpha_2 = 0.735
             self.wids = defaultdict(lambda: len(self.wids))

     def get_vocab_size(self):
          return len(self.wids)
	
     def store_counts(self, file):
              self.get_ngrams(file, self.order)
              if self.print_flag == 1:
		 print "Unique Unigrams like :", list(self.unigrams)[0], " are ", len(self.unigrams)
  		 print "Unique Bigrams like :", list(self.bigrams)[0], " are ", len(self.bigrams)
                 
     def get_features_v1(self):
       # This is a basic version which returns wid of every word as feature and its likelihood as target
       self.training_data = []
       self.num_features = 1
       c = 1
       feature_vector = np.zeros(int(self.get_vocab_size()))   # One hot k
       print feature_vector
       for line in self.data_array_train:
	 line = line.split()	 
	 for word in line:
	   feature_vector = np.zeros(int(self.get_vocab_size()))   # One hot k
	   c = c + 1
	   wid = self.wids[word]
	   feature_vector[wid] = 1
	   if c % 1000 == 1:
	     print word, wid, feature_vector[wid] 
	   self.training_data.append(feature_vector)   
       return   
     
     def feature_function(self, ctxt):
       features = []
       features.append(self.calculate_wid(ctxt))
       return features
     
     #def get_likelihood(self, word):
       
     
     def calculate_wid(self, ctxt):
       return wids[ctxt]  
  
     
     def read_corpus(self, file):
       # for each line in the file, split the words and turn them into IDs like this:
       print file
       f = open(file)
       self.data_array_train = []
       for line in f:
          line = line.split('\n')[0]
          self.data_array_train.append(line)
          words = line.split()
          for word in words:
              wid = self.wids[word]

     def print_words(self):
        for wid in self.wids:
            print wid, self.wids[wid]
    
    
     def print_dicts(self):
             print "Printing unigrams"
             for k in self.unigrams:
	       print k,self.unigrams[k]
	     print "Printing bigrams"
	     for  k in self.bigrams:
	       print k, self.bigrams[k]
	       

     def save_dicts(self):
           with open('unigrams.pkl', 'wb') as f:
                 pickle.dump(self.unigrams, f, pickle.HIGHEST_PROTOCOL)

           with open('bigrams.pkl', 'wb') as f:
                 pickle.dump(self.bigrams, f, pickle.HIGHEST_PROTOCOL)  

     def load_dicts(self):
           with open('unigrams.pkl', 'rb') as f:
                 self.unigrams = pickle.load(f)

           with open('bigrams.pkl', 'rb') as f:
                 self.bigrams = pickle.load(f)
	       
     def get_counts(self, file):
              self.print_flag = 1
              self.store_counts(file)
     
     # Calcuates n grams from a line     
     def ngrams(self,line, n):
              lst = line.split()
              output = {}
              for i in range(len(lst) -n + 1):
	           g = ' '.join(lst[i:i+n])
                   output.setdefault(g, 0)
                   output[g] += 1
              return output 
	    
     def combine_dicts(self, base_dict, small_dict):
         for key in small_dict:
             if base_dict.has_key(key):
	         base_dict[key] = int(base_dict[key]) + int(small_dict[key])
	     else:
	         base_dict[key] = int(small_dict[key])
         return base_dict 
   
     # Calculates n grams from a file     
     def get_ngrams(self, file, count):
              f = open(file)
              for line in f:
		   line = '<s> ' + line.split('\n')[0] + ' </s>'
		   bigrams = self.ngrams(line,2)
		   self.bigrams = self.combine_dicts(self.bigrams,bigrams)
		   unigrams = self.ngrams(line,1)
		   self.unigrams = self.combine_dicts(self.unigrams,unigrams)
                   self.remove_singletons(self.unigrams)
                   
     def remove_singletons(self, base_dict):
               for key in base_dict:
                   #print key, base_dict[key]
                   if base_dict[key] < 2:
	           #print key
	              base_dict[key] = 0
	           return   
	              
     def eqn8(self, strng, print_flag):
               l = len(strng.split())
               if l == 1:
                   estimate = (1 - self.alpha_1) * (float(self.unigrams[strng]) / sum(self.unigrams.values())) + self.alpha_unk * np.exp(1e-7)
                   return estimate
               else:
                   c = strng
                   #strng = strng.split()[:-1]
                   #p = ' '.join(tk for tk in strng)
                   p = c.split()[-1]
                   if print_flag ==1:
		      print "Bigram is ", c
                      print "Unigram is ", p
                   #print (1 - self.alpha_2) * (float(self.bigrams[c]) / self.bigrams.values()))
		   #print 			       
                   estimate = (1 - self.alpha_2) * (float(self.bigrams[c]) / sum(self.bigrams.values())) + (1 - self.alpha_1) * (float(self.unigrams[p]) / sum(self.unigrams.values()))  + self.alpha_unk * np.exp(1e-7)
                   return estimate

     def get_file_perplexity(self, file):
           f = open(file)
           print_flag = 0
           self.num_sentences = 0
           self.num_oovs = 0
           self.num_words = 0
           self.logprob = 0
           arr = []
           for line in f:
	        line = line.split('\n')[0].lower()
	        #line = '<s> ' + line.split('\n')[0] + ' </s>'
	        ppl_sentence = self.get_sentence_perplexity(line,0)
	        if print_flag ==1:
	           print line, ppl_sentence, '\n'
	        arr.append(ppl_sentence)
	   #print np.mean(arr) 
	   log_arr = np.log(arr)
	   print log_arr
	   ml_corpus = -1.0 * np.sum(log_arr) * 1.0/len(arr)
	   print np.exp(ml_corpus)
	   
	   print 'Sentences: ', self.num_sentences
	   print 'Words: ', self.num_words
	   print 'OOVs: ' , self.num_oovs
	   print 'Log probability: ', self.logprob
	   self.perplexity = np.exp( -1.0 * self.logprob / ( self.num_words + self.num_sentences - self.num_oovs) * 2.71)  # SRILM constant
	   print "Perplexity over corpus is: ", self.perplexity     
	        
	        
     def get_sentence_perplexity(self, string, print_flag):
           #print len(string)
           num_tokens = 0
           num_oovs = 0
           if len(string.split()) < 2:
	      print "The Sentence you gave me is very short"
	      return -1.0 * np.log(self.eqn8(string,0) / 2)
	   else:
	      mle_sentence = np.log(1.0)
	      line =  string.split('\n')[0] + ' </s>'
	      length_line = len(string.split())
	      line = line.split()
	      b = 0
	      while b < len(line) - 1:
		if print_flag ==1:
		   print "The value of  b is ", b
		kv = line[b] + ' ' + line[b+1]
		if print_flag ==1:
		 print "I am looking for ", kv
		if line[b+1] == '</s>':
		  kv = line[b]
		if print_flag ==1:
		  print "I am looking for ", kv
        	if kv in self.bigrams and self.bigrams[kv] > 0:  
		  if print_flag ==1:
		    print "Found ",kv , " in bigrams"
		  mle = self.eqn8(kv,0)
		  length_gram = 2
		else:
		  if print_flag ==1:
		   print  "I did not find ", kv, " in bigrams" 
		  
		  kv = line[b]
		  if print_flag ==1:
		      print "Now, I am searching for ", kv
		  if kv in self.unigrams and self.unigrams[kv] > 0:
		    if print_flag ==1:
		      print "Found ",kv , " in unigrams"
		    mle = self.eqn8(kv,0)
		    length_gram = 1
		  else:
		    if print_flag ==1:
		       print  "I did not find ", kv, " in unigrams or it was a singleton. I think its an UNK" 		    
		    kv = line[b]
		    mle = self.alpha_unk * np.exp(1e-7)
		    length_gram = 1
		    num_oovs = num_oovs + 1
		b = b + length_gram
		num_tokens = num_tokens + 1 
		mle_sentence = mle_sentence + np.log(mle)
	        self.num_oovs += num_oovs
	      self.num_sentences += 1
	      self.num_words += length_line
	      self.logprob += mle_sentence
	      print_flag = 0
	      mle_sentence_old = mle_sentence 
	      
	      mle_sentence = mle_sentence * (- 1.0 / (length_line + 1 +1 - num_oovs )  )
	      ppl_sentence = np.exp(mle_sentence * 2.3)
	      if print_flag ==1:
	        print "MLE of sentence is ", mle_sentence_old, " and PPL of sentence is ", ppl_sentence, " number of words: ", length_line, " number of OOVs: "  , num_oovs
	        g = open('t','w')
	        g.write(string + '\n')
	        g.close()
	        cmd = 'ngram -lm ../data/en-de/01_srilm_bigram.model -ppl t'
	        os.system(cmd)
	        print '\n\n'
	      print_flag = 0  
	      
	      return ppl_sentence
		
	      
	      
	      
 
  	              
               
