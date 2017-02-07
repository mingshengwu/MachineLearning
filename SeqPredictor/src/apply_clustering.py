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

model = joblib.load('clustering.pkl')
stds1=np.load('std1.npy')
stds2=np.load('std2.npy')
ms1=np.load('mean1.npy')
ms2=np.load('mean2.npy')
pca = joblib.load('pca.pkl')

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def transform(v,m1,std1,m2,std2,pca,model):
    vec=[]
    for e,s,m in zip(v,std1,m1):
        vec.append((e-m)/s)
    vec=np.array(vec)
    if max(abs(vec))>3:
        return None
    else:
        v=pca.transform([vec])[0]
        vec=[]
        for e,s,m in zip(v,std2,m2):
            vec.append((e-m)/s)
        return model.predict(vec)[0]

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

print transform(data[0],ms1,stds1,ms2,stds2,pca,model)




