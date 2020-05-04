#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =======================================================================================================
# 
#       Name : create_network.py
#       Description: Create network in nethiex algorithm
#       Created by : Mili Biswas
#       Created on: 21.02.2020
#
#       Dependency : Must be called from run.py
#
#      Purpose : This script is for reading the corpus. This is needed
#                as a part of Master Thesis based on FashionBrain Taxonomy Enrichment
#
#
# ========================================================================================================




from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os

import networkx as nx
import nltk
import numpy as np
from nltk.corpus import stopwords
sw = stopwords.words("english")

data_type={}

def fileExist(file):
    if os.path.exists(file):
        return True
    return False

def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)


def filter_word(wrd_freq,topics,vocab):
    #Filter imporatnt words
    for v in topics:
        tmp = []
        for tl in topics[v]:
            tl = [w for w in tl if w in vocab]
            if len(tl) > 0:
                tmp.append(tl)
        topics[v] = tmp
    return topics
    
def compute_tfidf(topics,vocab):
    # Compute tf-idf.
    tf = dict((v, {}) for v in topics)
    for v in topics:
        for tl in topics[v]:
            for w in tl:
                if w in tf[v]:
                    tf[v][w] += 1
                else:
                    tf[v][w] = 1
    idf = dict((w, 0) for w in vocab)
    for w in vocab:
        cnt = 0
        for v in topics:
            if w in tf[v]:
                assert tf[v][w] > 0
                cnt += 1
        idf[w] = np.log(len(topics) / cnt)
    tf_idf = dict((v, {}) for v in topics)
    for v in topics:
        for tl in topics[v]:
            for w in tl:
                tf_idf[v][w] = tf[v][w] * idf[w]
    return tf_idf

def create_network(dir_path,tf_idf,topics):
    '''
        Algorithm to 
    
    '''
    g = nx.Graph()
    for v in topics:
        scores = sorted(tf_idf[v].values())
        thresh = scores[-int(0.25 * len(scores))] # This is for top 5% from tfidf score
        for tl in topics[v]:
            for x in tl:
                if tf_idf[v][x] >= thresh:
                    for y in tl:
                        if x != y and tf_idf[v][y] >= thresh:
                            g.add_edge(x, y)
                            
    g = next(connected_component_subgraphs(g))
    
    mapping = dict(zip(g, range(g.number_of_nodes())))
    
    with open(os.path.join(dir_path,'db.voc'), 'w',encoding='iso-8859-1') as fout:
        for w, i in mapping.items():
            fout.write('%d %s\n' % (i, w))
            
    g = nx.relabel_nodes(g, mapping)
    
    elist_path=os.path.join(dir_path,'network.txt')
    
    nx.write_edgelist(g, elist_path, data=False)
    
    
def main_title(inputData,dirPath):
    
    if fileExist(os.path.join(dirPath,'db.elist.ncrp')):
        os.remove(os.path.join(dirPath,'db.elist.ncrp'))
        
    if fileExist(os.path.join(dirPath,'db.voc')):
        os.remove(os.path.join(dirPath,'db.voc'))
        
    if fileExist(os.path.join(dirPath,'network.txt')):
        os.remove(os.path.join(dirPath,'network.txt'))
    
    
    
    wrd_freq = {}
    topics = {}
    corpus=inputData
    for key,paper in corpus.items():
        for tl in paper:
            tl = nltk.word_tokenize(tl)
            tl = [w.lower() for w in tl]
            for w in tl:
                if w in wrd_freq:
                    wrd_freq[w] += 1
                else:
                    wrd_freq[w] = 0
            if key not in topics:
                topics[key] = []
            topics[key].append(tl)
    
    vocab = set([w for w in wrd_freq])
    topics=filter_word(wrd_freq,topics,vocab)
    tf_idf=compute_tfidf(topics,vocab)
    
    create_network(dirPath,tf_idf,topics)
    
if __name__ == '__main__':
    if len(sys.argv[1:])==2:
        inputData=sys.argv[1]
        dirPath=sys.argv[2]
        np.random.seed(0)
        main_title(inputData,dirPath)
    else:
        sys.exit('[Error]: Nethiex algorithm has been given with wrong number of parameters, where 2 parameters are expected')
