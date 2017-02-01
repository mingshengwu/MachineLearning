__author__ = 'billywu'


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA


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
print metrics.silhouette_score(X,kmeans.labels_, metric='euclidean')

for i in range(10):
    plt.scatter(centers[i,0],centers[i,1],color='red')
plt.show()

new_data = genfromtxt('../data/anomaly/allfields_anom.txt', delimiter=',')
new_X=new_data[:,1:]
new_X=pca.transform(new_X)

y=kmeans.predict(new_X)
print y[0:288]
print y[288:480]