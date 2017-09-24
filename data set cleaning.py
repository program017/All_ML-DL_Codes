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
lib=nltk.FreqDist('')   #creating blank freqency distribution
loadfile=open('Enron.test','r')     
for line in loadfile:
    line=line.split()   
    label=int(line[0].strip(',')[-1])   # removing ',' if present
    if label==1:                    
        data=wt((' ').join(line[2:]).lower())   
        for words in data:
            words=lemmatizer.lemmatize(words,pos='a')   
            if words not in stop_words:         
                lib[words]+=1
print 'time taken:{}'.format(time.time()-t1)        
