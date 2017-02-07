__author__ = 'billywu'

import numpy as np
import re
import csv
import os
from os import listdir

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

filenames = find_csv_filenames("../data/ips/")

for f in filenames:
    os.remove("../data/ips/"+f)

with open('../data/ml_test0201_ports/fluency_201701_ipSummary.json') as f:
    content = f.readlines()

header=['rxB', 'txB', 'totalB', 'flowCount', 'start_ms', 'dur', 'fanOut', 'fanIn']
records=[]
files=[]
ip=None
new=True
for c in content:
    line=c.split('},{')
    for p in line:
        record=[]
        raw=p.split(':{')[1].split(",")
        for pp in raw:
            temp=re.findall(r'\b\d+\b', pp)
            if len(temp)==1:
                record.append(int(temp[0]))
            else:
                ip=temp[0]+'.'+temp[1]+'.'+temp[2]+'.'+temp[3]
                if ip not in files:
                    files.append(ip)
                    new=True
                else:
                    new=False
        if new:
            fd = open('../data/ips/'+ip+'.csv','w')
            writer = csv.writer(fd)
            #writer.writerow(header)
            writer.writerow(record)
            fd.close()
        else:
            fd = open('../data/ips/'+ip+'.csv','a')
            writer = csv.writer(fd)
            writer.writerow(record)
            fd.close()
