import time
import nltk
from nltk.tokenize import word_tokenize as wt
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import WordNetLemmatizer

t1=time.time()      #for checking run time
lemmatizer=WordNetLemmatizer()      #creating lemmatizer
stop_words=list(set(stopwords.words("english")))    #creating english stop words
stop_words.append('.');stop_words.append(',');  #adding , . to stop words
testloadfile=open('Enron.train','r')
testwritefile=open('enrontestfile','w')
for line in testloadfile:
    line=line.split()   
    label=line[0].strip(',')
    testwritefile.write(label)
    testwritefile.write(': ') 
    data=wt((' ').join(line[2:]).lower())
    lib=nltk.FreqDist('')
    for words in data:
        words=lemmatizer.lemmatize(words,pos='a')   
        if words not in stop_words:
            lib[words]+=1
    for i in lib.most_common(len(lib)):     #not to write same word again
        testwritefile.write(i[0])
        testwritefile.write(' ')        #seperating by space in final file
    print >>testwritefile
print 'time taken:{}'.format(time.time()-t1)        
