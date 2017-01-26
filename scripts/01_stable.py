
from itertools import tee, islice
from collections import Counter
import  re

file_name = '../data/en-de/train.en-de.en'
#file_name = 'test'


# String and length of ngram needed
def store_ngrams(s, n):
  tlst = s
  while True:
    a, b = tee(tlst)    
    l = tuple(islice(a, n))
    if len(l) == n:
      yield l
      next(b)
      tlst = b
    else:
      break
    
def ngrams(lst, n):
  lst = lst.split(' ')
  output = {}
  for i in range(len(lst)-n+1):
    g = ' '.join(lst[i:i+n])
    output.setdefault(g, 0)
    output[g] += 1
  return output
    
n = 2    
k = {}    
f = open(file_name)
for line in f:
  line = '<s> ' + line.split('\n')[0] + ' </s>'
  #print "The line is " + str(line)
  t =  ngrams(line,n)
  #print t
  for ke in t:
    #print ke
    if k.has_key(ke):
       k[ke] = int(k[ke]) + int(t[ke])
    else:
       k[ke] = int(t[ke])

#strng = "thank you"
#print k[strng]
  
  
