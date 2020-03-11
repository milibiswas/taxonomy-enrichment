##########################################################################################
#
#  Script Name :
#  Created By  : Mili Biswas (MSc, Computer Science, UNIFR)
#  Creation Date:
#
#  Purpose : This script is for reading the corpus. This is needed
#            as a part of Master Thesis based on FashionBrain Taxonomy Enrichment
#
##########################################################################################


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle
import xml.etree.ElementTree

#from matplotlib import pyplot as plt

import networkx as nx
import nltk
import numpy as np

from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
sw = stopwords.words("english")
ps=PorterStemmer()

data_type={}

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
    g = next(nx.connected_component_subgraphs(g))
    mapping = dict(zip(g, range(g.number_of_nodes())))
    with open(os.path.join(dir_path,'db.voc'), 'w',encoding='iso-8859-1') as fout:
        for w, i in mapping.items():
            fout.write('%d %s\n' % (i, w))
    g = nx.relabel_nodes(g, mapping)
    elist_path=os.path.join(dir_path,'network.txt')
    nx.write_edgelist(g, elist_path, data=False)
    
    #------------ plot -------------
    
    '''pos = nx.spring_layout(g)
    plt.figure(figsize=(20,25))
    nx.draw(g, pos, edge_color='k', node_color='w' ,with_labels=True, font_weight='light', width= 0.5)
    plt.savefig("fashion.png")
    plt.show()'''
    
    
def readData(file_path):
    blog_dict= {}
    strVal=''
    key=''
    with open(file_path,'r',encoding='iso-8859-1')as fin:
        for cnt, line in enumerate(fin):
            if line.strip():                     # This indicates no blank lines will be processed
                if len(line.strip().split())>2:  # This identifies block start
                    _,_,z=line.strip().split()  

                    '''
                        Here check if previous block has been read. If yes, append that in the blog_dict
                    '''

                    key = z
                    if key not in blog_dict:
                        blog_dict[key]=[]

                else:

                    '''
                          bg.,n This part prepares the string value to be inserted in blog_dict

                    '''

                    x,y=line.strip().split()
                    word_type=y.strip()
                    if ((word_type.lower()=='e-item')):
                        strVal = strVal+x+' '
            else:
                if strVal:
                    blog_dict[key].append(strVal.rstrip())
                    strVal=''
    return blog_dict
        
    
def main_title(dir_path="./data/"):
    wrd_freq = {}
    topics = {}
    corpus=readData(os.path.join(dir_path,'data_blog_id.txt'))
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
    create_network(dir_path,tf_idf,topics)
    
if __name__ == '__main__':
    np.random.seed(0)
    main_title()
