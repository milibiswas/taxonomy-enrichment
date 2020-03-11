# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans

def cluster(prefix='./data/',levels=[0] * 1 + [1] * 5):
    
    lvl_cat=[(0,1),(1,5)]
    
    with open(prefix + 'db.elist.ncrp') as fin:
        n, d = fin.readline().split()
        n, d = int(n), int(d)
        #tree_sz = len(levels)
        tree_lv = max(levels) + 1
        d_node = d // (tree_lv + 1)
        #print(n, d)
        feats = np.zeros((n, d), dtype=np.float64)
        for line in fin:
            line = line.split()
            assert len(line) == d + 1
            feats[int(line[0])] = [float(x) for x in line[1:]]
    
    fileObj=open("./data/db.ncrp.cls","w", encoding="utf-8")
    cnt=0
    for lv,n_cat in lvl_cat:
        word_m=feats[:, (d_node * lv):(d_node * (lv + 1))]
        kmeans = KMeans(n_clusters=n_cat, random_state=0).fit(word_m)
        XB=kmeans.cluster_centers_
        if lv>0:
            for i_cat in range(n_cat):
                str1=str(cnt)
                for elem in list(XB[i_cat,:]):
                    str1=str1+" "+str(elem)
                fileObj.write(str1+"\n") 
                cnt += 1    
        else:
            str1=str(cnt)
            for elem in list(XB[0,:]):
                str1=str1+" "+str(elem)
            fileObj.write(str1+"\n") 
            cnt += 1
    
    
    
if __name__ == "__main__":
    cluster()
