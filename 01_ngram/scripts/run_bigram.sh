for f in '../data/en-de/train.en-de.'* ; do
 echo $f
 python bigram_lm_v2.py $f
done 

