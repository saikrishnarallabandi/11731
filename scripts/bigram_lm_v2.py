
from itertools import tee, islice
from collections import Counter
import  re, math
import numpy as np
from math import exp

file_name = '../data/en-de/train.en-de.en'
test_file = '../data/en-de/test.en-de.en'

alpha_unk = 0.02
alpha_1 = 0.245
alpha_2 = 0.735

def eqn8(strng):
  l = len(strng.split())
  if l == 1:
    estimate = (1 - alpha_1) * (float(unigram_dict[strng]) / sum(unigram_dict.values())) + alpha_unk * exp(1e-7)
    return estimate
  else:
    c = strng
    strng = strng.split()[:-1]
    p = ' '.join(tk for tk in strng)
    estimate = (1 - alpha_2) * (float(bigram_dict[c]) / unigram_dict[p]) + (1 - alpha_1) * (float(unigram_dict[p]) / sum(unigram_dict.values()))  + alpha_unk * exp(1e-7)
    return estimate
  
def corpus_perplexity(arr):  
  a = sum(np.log(arr))
  b = -1.0 
  c = len(arr)
  print np.exp(a * b / c)
  
def combine_dicts(base_dict, small_dict):
     for key in small_dict:
         if base_dict.has_key(key):
	   base_dict[key] = int(base_dict[key]) + int(small_dict[key])
	 else:
	   base_dict[key] = int(small_dict[key])
     return base_dict 
   
def remove_singletons(base_dict):
     for key in base_dict:
        #print key, base_dict[key]
        if base_dict[key] == 1:
	  #print key
	  base_dict[key] = 0

def get_ngrams(lst, n):
      lst = lst.split()
      output = {}
      for i in range(len(lst) -n + 1):
	    g = ' '.join(lst[i:i+n])
            output.setdefault(g, 0)
            output[g] += 1
      return output

n = 2
bigram_dict = {}
unigram_dict = {}

f = open(file_name)
for line in f:
  line = '<s> ' + line.split('\n')[0] + ' </s>'
  #line =  line.split('\n')[0] 
  bigrams = get_ngrams(line,2)
  bigram_dict = combine_dicts(bigram_dict,bigrams)
  unigrams = get_ngrams(line,1)
  unigram_dict = combine_dicts(unigram_dict,unigrams)

print "Populated bigram and unigrams"

remove_singletons(unigram_dict)
print "Removed unigram singletons"
remove_singletons(bigram_dict)
print "Removed bigram singletons"

strng = "you know"
mle = eqn8(strng)
print mle  
ppl = exp(-(math.log(mle)/len(strng.split())))
print ppl


ppl_array = []  
MLE = []
LENGTH_LINE =[]
terms = 0
words = []
f = open(test_file)
count = 0
############################ WITHIN CORPUS #######################
for line in f:
 if len(line) < 2:
   pass
 else:
  ########################## WITHIN SENTENCE ###################### 
  #print line
  words.append(w for w in line)
  count = count + 1
  mle_sentence = np.log(1.0) 
  line = line.split('\n')[0] + ' </s>'
  line = line.split()
  length_line = len(line) -1
  LENGTH_LINE.append(length_line)
  #print line
  b = 0
  while b < len(line) -1:  
    
    kv = line[b] + ' ' + line[b+1]
    # Eliminate the stop symbol from calculation
    if line[b+1] == '</s>':
      kv = line[b]
    
    if kv in bigram_dict and bigram_dict[kv] > 0:
      #print kv
      mle = eqn8(kv)
      length_string = 2
    else:
      kv = line[b]
      if kv in unigram_dict and unigram_dict[kv] > 0:
	#print kv
	mle = eqn8(kv)
	length_string = 1
      else:
	#print line[b]
	kv  = line[b]
        mle = alpha_unk * exp(1e-7)
        length_string = 1
    #print kv    
    b = b + length_string
    terms = terms + 1
    #mle = mle / length_line
    #mle = mle / terms
    mle_sentence = mle_sentence + np.log(mle)
    #print mle_sentence
  #print '\n\n\n'
  #MLE.append(mle_sentence / terms)
 
  mle_sentence = np.exp(-1.0 * mle_sentence/length_line)
  print mle_sentence
  MLE.append(mle_sentence)
  #print sum(MLE), sum(LENGTH_LINE) , sum(MLE) / sum(LENGTH_LINE)
#ppl = exp(-(sum(MLE))/sum(LENGTH_LINE))
#ppl = exp(-1/len(MLE) * (np.sum(MLE)))  
corpus_perplexity(MLE)
  #ppl = exp(1-(sum(MLE)))
#print mean(MLE)
#ppl_array.append(ppl)

#print np.mean(ppl_array)    
	