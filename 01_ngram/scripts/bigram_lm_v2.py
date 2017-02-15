
from itertools import tee, islice
from collections import Counter
import  re, math, sys
import numpy as np
from math import exp

file_name = sys.argv[1]
test_file = '../data/en-de/test.en-de.en'
test_file = file_name.split('/train')[0] + '/test' + file_name.split('/train')[1]
print test_file
#test_file = 't'

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
  print "Perplexity of this : ", len(arr) , "  length sentence is : " , np.power(10, (a * b / c))
  #corpus_perplexity_v2(arr)

def corpus_perplexity_v3(arr, l):  
  a = sum(np.log2(arr))
  b = -1.0 
  c = l
  print "Perplexity: " , np.power(2, (a * b / c))
  
def corpus_perplexity_v2(arr):
  p = 1.0
  for a in arr:
    p = p * a
  lp = np.log2(p)
  print np.exp(-1.0 * lp / len(arr))

  
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
        if base_dict[key] < 2:
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

n = 1
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

#print "Populated bigram and unigrams"

unigram_dict_original = unigram_dict
print "total unigrams: ", sum(unigram_dict.values())
print "Number of unigrams: ", len(unigram_dict)
#remove_singletons(unigram_dict)
#print "Removed unigram singletons"
bigram_dict_original = bigram_dict
remove_singletons(bigram_dict)
#print "Removed bigram singletons"


ppl_array = []  
MLE = []
PROB_SENTENCE = []
SENTENCE_LENGTH = 0
LENGTH_LINE =[]
terms = 0
words = []
f = open(test_file)
count = 0
############################ WITHIN CORPUS #######################
for line in f:
 line = line.split('\n')[0] 
 if len(line) < 2:
   pass
 else:
  ########################## WITHIN SENTENCE ###################### 
  print line
  words.append(w for w in line)
  count = count + 1
  mle_sentence = np.log2(1.0) 
  prob_sentence = 1.0
  line =  line.split('\n')[0] + ' </s>'
  line = line.split()
  print "Length  is ", len(line)

  length_line = len(line) 
  LENGTH_LINE.append(length_line)
  
  b = 0
  while b < len(line) -1:  
    
    kv = line[b] + ' ' + line[b+1]
    # Eliminate the stop symbol from calculation
    if line[b] == '</s>':
      kv = line[b]
       #pass
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
    #print b
    terms = terms + 1
    mle = mle / length_line
    #mle = mle / terms
    mle_sentence = mle_sentence + np.log2(mle)
    prob_sentence = prob_sentence * mle
    #print prob_sentence
    SENTENCE_LENGTH = SENTENCE_LENGTH + length_line
    #print mle_sentence
  #print '\n\n\n'
  #MLE.append(mle_sentence / terms)
  
  #ppl_sentence_v1 =  1/(pow(prob_sentence, 1.0/length_line))
  ppl_sentence = np.power(2, (-1.0 * mle_sentence/length_line))
  print " Perplexity of sentence is : ", ppl_sentence#, ppl_sentence_v1, prob_sentence
  
  MLE.append(ppl_sentence)
  PROB_SENTENCE.append(prob_sentence)
  #print sum(MLE), sum(LENGTH_LINE) , sum(MLE) / sum(LENGTH_LINE)
#ppl = exp(-(sum(MLE))/sum(LENGTH_LINE))
#ppl = exp(-1/len(MLE) * (np.sum(MLE))) 
print "MLE ", MLE
#corpus_perplexity(MLE)
corpus_perplexity_v3(PROB_SENTENCE, SENTENCE_LENGTH)
print '\n\n'
print "Mean is  ", np.mean(MLE)
  #ppl = exp(1-(sum(MLE)))
#print mean(MLE)
#ppl_array.append(ppl)

print np.mean(ppl_array)    
	