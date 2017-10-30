import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import matplotlib.pyplot as plt
df_list = []
with open('enrontestfile') as f:
    for line in f:
        
        line = line.strip()
        
        columns = re.split(':', line, maxsplit=4)
        df_list.append(columns)
df = pd.DataFrame(df_list)
#print df[0]
#print df[1]
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df[1])
#print X_train_counts
#print X_train_counts.shape
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
#print X_train_tf.shape

#df[0].target[:10]
y=[]
for i in range(len(df)):
    temp=df[0][i]
    y.append(temp[-1])
Y=np.array(y)

#Import Library
from sklearn.ensemble import RandomForestClassifier #use Rando


####         training is done       #####
df_list = []
with open('enrontest') as f:
    for line in f:
        line = line.strip()
        columns = re.split(':', line, maxsplit=4)
        df_list.append(columns)
docs_new = pd.DataFrame(df_list)
y_test=[]
for i in range(len(docs_new)):
    temp=df[0][i]
    y_test.append(temp[-1])
Y_test=np.array(y_test)

astimators_list=[]
accuracy_list=[]
for i in range(1,20):
    
    model= RandomForestClassifier(n_estimators=i)

    #print 'traning'
    model.fit(X_train_tf,Y)

    X_test_counts = count_vect.transform(docs_new[1])
    tf_transformer = TfidfTransformer(use_idf=False)
    X_new_tfidf = tf_transformer.transform(X_test_counts)

    predicted= model.predict(X_new_tfidf)
    #print predicted
    from sklearn import metrics
    # testing score
    crosstab=pd.crosstab(predicted,Y_test, rownames=['Actual'], colnames=['Predicted'])
    #score = metrics.f1_score(Y_test, predicted,pos_label=Y_test)
    # training score
    #score_train = metrics.f1_score(y_train, pred_train, pos_label=list(set(y_train)))
    print crosstab
    correct=crosstab['1']['1']+crosstab['0']['0']
    total=crosstab['1']['1']+crosstab['0']['0']+crosstab['0']['1']+crosstab['1']['0']
    accuracy=float(correct)/float(total)
    print "accuracy:{}".format(accuracy*100)
    astimators_list.append(i)
    accuracy_list.append(accuracy)
    #print score


plt.plot(astimators_list,accuracy_list,color='b')
plt.scatter(astimators_list,accuracy_list,color='b',marker='x',s=50)
plt.xlabel('estimators')
plt.ylabel('accuracy')
#plt.legend()
plt.title("accuracy vs estimators")
plt.grid()
plt.show()
