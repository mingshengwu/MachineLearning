__author__ = 'billywu'


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim

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

model.save_word2vec_format('State_Representation')
sentence=np.array(sentence)
np.save('train',sentence)

