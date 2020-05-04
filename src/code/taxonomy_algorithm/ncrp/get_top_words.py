from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pickle as pc

def main(prefix='./data/', levels=[0] * 1 + [1] * 5):
    # Read vertex representations.
    with open(prefix + 'db.elist.ncrp',encoding='iso-8859-1') as fin:
        n, d = fin.readline().split()
        n, d = int(n), int(d)
        #print(n, d)
        feats = np.zeros((n, d), dtype=np.float64)
        for line in fin:
            line = line.split()
            assert len(line) == d + 1
            feats[int(line[0])] = [float(x) for x in line[1:]]

    # Read cluster representations.
    tree_sz = len(levels)
    tree_lv = max(levels) + 1
    d_node = d // (tree_lv + 1)
    #print(tree_sz, tree_lv, d_node)
    clust = np.zeros((tree_sz, d_node), dtype=np.float64)
    with open(prefix + 'db.ncrp.cls',encoding='iso-8859-1') as fin:
        cnt = 0
        for line in fin:
            cnt += 1
            line = line.split()
            assert len(line) == d_node + 1
            clust[int(line[0])] = [float(x) for x in line[1:]]
        assert cnt == tree_sz

    # Read vocabulary.
    id2wd = ['-1' for _ in range(n)]
    with open(prefix + 'db.voc',encoding='iso-8859-1') as fin:
        cnt = 0
        for line in fin:
            cnt += 1
            i, wd = line.split()
            i, wd = int(i), wd.strip()
            id2wd[i] = wd

    # Find the core members & dump that in a file.

    cat_dict={}

    cat_file_obj=open("./data/cat_file.dict","wb")


    for t in range(tree_sz):
        lv = levels[t]
        crep = clust[t].reshape(1, d_node)
        xrep = feats[:, (d_node * lv):(d_node * (lv + 1))]
        dist = np.sum((xrep - crep) ** 2, axis=1)
        assert dist.shape == (n,)
        min_idx = np.argsort(dist)[:10]
        print(t, end=' : ')
        strVal=""
        for i in min_idx:
            print(id2wd[i], end='_')
            strVal=strVal+"_"+id2wd[i]
        cat_dict[t]=strVal
        print()
    pc.dump(cat_dict,cat_file_obj)
    cat_file_obj.close()

if __name__ == '__main__':
    np.random.seed(0)
    main()
