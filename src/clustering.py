__author__ = 'billywu'


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
import gensim
import tensorflow as tf
from scipy import spatial

from numpy import genfromtxt
my_data = genfromtxt('../data/normal/allfields.txt', delimiter=',')

X=my_data[:,1:]
pca = PCA(n_components=2)
pca.fit(X)
X=pca.transform(X)
ms=[]
stds=[]
for i in range(len(X[0,:])):
    m=np.mean(X[:,i])
    std=np.std(X[:,i])

    my_data[:,i]=(X[:,i]-m)/std

plt.scatter(X[:,0],X[:,1])
kmeans = KMeans( n_clusters=10, init='k-means++',random_state=0).fit(X)
centers=kmeans.cluster_centers_

sentence=[]
for l in kmeans.labels_:
    sentence.append(str(l))
sentences=[]
for i in range(1000):
    sentences.append(sentence)
model = gensim.models.Word2Vec(sentences, size=100,min_count=1)

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

def next(model,l,sess, pred):
    input=np.hstack((model[l[0]],model[l[1]],model[l[2]],model[l[3]],model[l[4]]))
    v=sess.run(pred, feed_dict={x:[input]})
    print v
    min=1000
    amin=0
    for i in range(10):
        if min>np.inner(v-model[str(i)],v-model[str(i)]):
            min=np.inner(v-model[str(i)],v-model[str(i)])
            amin=i
    return amin,min

with tf.Session() as sess:
    sess.run(init)

    sess.run(init)
    for epoch in range(300):
        _, c = sess.run([optimizer, cost], feed_dict={x: dataX, y: dataY})
        print("Epoch:", '%04d' % (epoch+1), "cost=",
                c)
    print("Optimization Finished!")
    print next(model,['1','9','9','9','1'],sess,pred)
    print next(model,['9','9','9','1','6'],sess,pred)




