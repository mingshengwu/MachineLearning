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
import gensim
import matplotlib.pyplot as plt
import pickle

in_model = joblib.load('model/inner/cleanned_clustering.pkl')
in_stds1=np.load('model/inner/all_std.npy')
in_stds2=np.load('model/inner/cleaned_std2.npy')
in_ms1=np.load('model/inner/all_mean.npy')
in_ms2=np.load('model/inner/cleaned_mean.npy')
in_pca = joblib.load('model/inner/pca_cleaned.pkl')

out_model = joblib.load('model/outter/cleanned_clustering.pkl')
out_stds1=np.load('model/outter/all_std.npy')
out_stds2=np.load('model/outter/cleaned_std2.npy')
out_ms1=np.load('model/outter/all_mean.npy')
out_ms2=np.load('model/outter/cleaned_mean.npy')
out_pca = joblib.load('model/outter/pca_cleaned.pkl')

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def transform(v,in_m1,in_std1,in_m2,in_std2,in_pca,in_model,out_m1,out_std1,out_m2,out_std2,out_pca,out_model):
    vec=[]
    for e,s,m in zip(v,in_std1,in_m1):
        vec.append((e-m)/s)
    vec=np.array(vec)
    if max(abs(vec))>3:
        v=out_pca.transform([vec])[0]
        vec=[]
        for e,s,m in zip(v,out_std2,out_m2):
            vec.append((e-m)/s)
        return "out"+str(out_model.predict(vec)[0])
    else:
        v=in_pca.transform([vec])[0]
        vec=[]
        for e,s,m in zip(v,in_std2,in_m2):
            vec.append((e-m)/s)
        return "in"+str(in_model.predict(vec)[0])

filenames = find_csv_filenames("../data/ips/")

data=[]
for f in filenames:
    my_data = genfromtxt("../data/ips/"+f, delimiter=',')
    if not my_data.shape==(8,):
        my_data=np.delete(my_data, 4, axis=1)
        data.append(my_data.tolist())
    else:
        my_data=np.delete(my_data,4,axis=0)
        data.append([my_data.tolist()])

sents=[]
print "Representation Training Started..."
for d in data:
    ret=[]
    for dd in d:
        ret.append(transform(dd,in_ms1,in_stds1,in_ms2,in_stds2,in_pca,in_model,
                        out_ms1,out_stds1,out_ms2,out_stds2,out_pca,out_model))
    if len(ret)>=2:
        sents.append(ret)

model = gensim.models.Word2Vec(sents, size=100,min_count=1)
model.train(sents)
model.train(sents)
model.train(sents)
model.train(sents)
model.train(sents)
model.train(sents)
model.save_word2vec_format("Representation",binary=False)
print "Representation Training Finished"



