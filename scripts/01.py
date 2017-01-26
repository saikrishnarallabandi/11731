
file_name = '../data/en-de/train.en-de.en'

unigram_array = []
bigram_array = []
trigram_array = []
f = open(file_name)
for line in f:
  line = '<s> ' + line.split('\n')[0] + ' </s>'
  line = line.split()
  #print line
  k = 0
  for k in range(len(line)):
    try:
      word,n_word,nn_word = line[k], line[k+1], line[k+2]
      unigram_array.append(word)
      bigram_array.append(word + ' ' + n_word)
      trigram_array.append(word + ' ' + n_word + ' ' + nn_word)
    except IndexError:
      unigram_array.append(line[k])
      try: 
         bigram_array.append(line[k] + ' ' + line[k+1])
      except IndexError:
	 pass
    k = k + 1

#print unigram_array    
  
  
  
