from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split as tts
from sklearn.linear_model import LogisticRegressionCV as lrcv
from sklearn.linear_model import LinearRegression as lr
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation

iris =load_iris()

x,y=iris.data[:,:4],iris.target
train_x,test_x,train_y,test_y = tts(x,y,train_size=0.5, random_state=0)


lrg=lr()
lrg.fit(train_x,train_y)
pred_y=lrg.predict(test_x)

print("accuracy lr: {}".format(lrg.score(test_x,test_y)))
#keras model

model=Sequential()
model.add(Dense(16,input_shape=(4,)))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('sigmoid'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(train_x,train_y,verbose=1,batch_size=1,nb_epoch=100)
loss,accuracy=model.evaluate(test_x,test_y,verbose=0)
#print("accuracy keras: {.2f}".format(accuracy))

print("accuracy keras: {}".format(accuracy))
print("accuracy lr: {}".format(lrg.score(test_x,test_y)))
