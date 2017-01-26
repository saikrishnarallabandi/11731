
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
  
def combine_dicts(base_dict, small_dict):
     for key in small_dict:
         if base_dict.has_key(key):
	   base_dict[key] = int(base_dict[key]) + int(small_dict[key])
	 else:
	   base_dict[key] = int(small_dict[key])
     return base_dict 

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


strng = "you know"
mle = eqn8(strng)
print mle  
ppl = exp(1-(math.log(mle)/len(strng.split())))
print ppl  








ppl_array = []  
MLE = []
LENGTH_LINE =[]
words = []
f = open(file_name)
for line in f:
 if len(line) < 2:
   pass
 else:
  #print line
  mle_sentence = math.log(1.0) 
  line = line.split('\n')[0] + ' </s>'
  line = line.split()
  words.append(w for w in line)
  length_line = len(line) -1
  LENGTH_LINE.append(length_line)
  #print line
  b = 0
  while b < len(line) -1:
    
    kv = line[b] + ' ' + line[b+1]
    if kv in bigram_dict:
      #print kv
      mle = eqn8(kv)
      length_string = 2
    else:
      kv = line[b]
      if kv in unigram_dict:
	#print kv
	mle = eqn8(kv)
	length_string = 1
      else:
	kv  = line[b]
        mle = alpha_unk * exp(1e-7)
        length_string = 1
    b = b + length_string
    #mle = mle / length_string
    mle_sentence = mle_sentence + math.log(mle)
    #print mle_sentence
  MLE.append(mle_sentence)
print sum(MLE), sum(LENGTH_LINE) , sum(MLE) / sum(LENGTH_LINE)
ppl = exp(-(sum(MLE))/sum(LENGTH_LINE))
  #ppl = exp(1-(sum(MLE)))
print  ppl
#ppl_array.append(ppl)

#print np.mean(ppl_array)    
	