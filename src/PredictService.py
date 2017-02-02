__author__ = 'billywu'

import tensorflow as tf
import numpy as np
import gensim

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



def next(model,l,sess, pred):
    input=np.hstack((model[l[0]],model[l[1]],model[l[2]],model[l[3]],model[l[4]]))
    v=sess.run(pred, feed_dict={x:[input]})
    min=1000
    amin=0
    for i in range(10):
        if min>np.inner(v-model[str(i)],v-model[str(i)]):
            min=np.inner(v-model[str(i)],v-model[str(i)])
            amin=i
    return amin


sess = tf.Session()
param=np.load('ModelParam.npy')
for t,p in zip(tf.all_variables(),param):
    sess.run(t.assign(p))
print next(model,['1','9','9','9','1'],sess,pred)
