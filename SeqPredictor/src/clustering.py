__author__ = 'billywu'

import numpy as np
import re
import csv
import os
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import cluster
from os import listdir
from numpy import genfromtxt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pickle


def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

filenames = find_csv_filenames("../data/ips/")

data=[]
for f in filenames:
    my_data = genfromtxt("../data/ips/"+f, delimiter=',')
    if not my_data.shape==(8,):
        my_data=np.delete(my_data, 4, axis=1)
        data=data+my_data.tolist()
    else:
        my_data=np.delete(my_data,4,axis=0)
        data.append(my_data.tolist())
X=np.array(data)
ms=[]
stds=[]
for i in range(len(X[0,:])):
    m=np.mean(X[:,i])
    std=np.std(X[:,i])
    X[:,i]=(X[:,i]-m)/std
    ms.append(m)
    stds.append(std)

ms=np.array(ms)
stds=np.array(stds)
np.save('mean1',ms)
np.save('std1',stds)

cleaned_data=[]
for l in X:
    if max(abs(l))<3:
        cleaned_data.append(l)
X=np.array(cleaned_data)
cleaned_data=np.array(cleaned_data)
pca = PCA(n_components=5)
pca.fit(X)
joblib.dump(pca, 'pca.pkl')
X=pca.transform(X)
ms=[]
stds=[]
for i in range(len(X[0,:])):
    m=np.mean(X[:,i])
    std=np.std(X[:,i])
    X[:,i]=(X[:,i]-m)/std
    ms.append(m)
    stds.append(std)
np.save('mean2',ms)
np.save('std2',stds)
X=X[np.random.choice(X.shape[0], 10000, replace=False), :]
for k in range(4,9):
    model = KMeans( n_clusters=k, init='k-means++',random_state=0).fit(X)
    #model = MiniBatchKMeans(init='k-means++', n_clusters=11, batch_size=100,
    #                  n_init=10, max_no_improvement=10, verbose=0).fit(X)
    score=silhouette_score(X,model.labels_)
    print k, score
if score>0.59:
    joblib.dump(model, 'clustering.pkl')
else:
    print "Clustering not good enough"
#plt.scatter(X[model.labels_==1,0],X[model.labels_==1,1],color='red')
#plt.scatter(X[model.labels_==0,0],X[model.labels_==0,1],color='blue')
#plt.scatter(X[model.labels_==2,0],X[model.labels_==2,1],color='green')
#plt.scatter(X[model.labels_==3,0],X[model.labels_==3,1],color='yellow')
#plt.scatter(X[model.labels_==4,0],X[model.labels_==4,1],color='purple')
#plt.scatter(X[model.labels_==5,0],X[model.labels_==5,1],color='cyan')
#plt.scatter(X[model.labels_==6,0],X[model.labels_==6,1],color='black')
#plt.scatter(X[model.labels_==7,0],X[model.labels_==7,1],color='m')
#plt.scatter(X[model.labels_==8,0],X[model.labels_==8,1],color='#eeefff')
#plt.show()
