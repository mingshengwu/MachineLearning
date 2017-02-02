__author__ = 'billywu'

import tensorflow as tf
import gensim
import numpy as np

sentence=np.load('train.npy').tolist()
model=gensim.models.Word2Vec.load_word2vec_format('State_Representation')

x = tf.placeholder(tf.float32, [None, 5*100])
y = tf.placeholder(tf.float32, [None, 100])
w1=tf.Variable(tf.random_normal([5*100, 2000]))
w4=tf.Variable(tf.random_normal([2000, 100]))

b1= tf.Variable(tf.random_normal([2000]))
b4= tf.Variable(tf.random_normal([100]))

layer1= tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
pred= tf.add(tf.matmul(layer1, w4), b4)
cost = tf.reduce_mean(tf.square(pred-y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
init = tf.initialize_all_variables()
input=[]
sentence1=np.array(sentence)
input.append(model[sentence.pop(0)])
input.append(model[sentence.pop(0)])
input.append(model[sentence.pop(0)])
input.append(model[sentence.pop(0)])
input.append(model[sentence.pop(0)])
dataX=[]
dataY=[]
while (sentence!=[]):
    dataX.append(np.hstack((input[0],input[1],input[2],input[3],input[4])))
    xx=sentence.pop(0)
    dataY.append(model[xx])
    input.pop(0)
    input.append(model[xx])
dataX=np.array(dataX)
dataY=np.array(dataY)



saver = tf.train.Saver(tf.all_variables())

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(300):
        _, c = sess.run([optimizer, cost], feed_dict={x: dataX, y: dataY})
        print("Epoch:", '%04d' % (epoch+1), "cost=",
                c)
    print("Optimization Finished!")
    np.save("ModelParam",sess.run(tf.all_variables()))





