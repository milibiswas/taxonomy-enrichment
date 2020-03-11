#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =======================================================================================================
# 
#       Name : keyword_extract.py
#       Description: To run the software package
#       Created by : Mili Biswas
#       Created on: 21.02.2020
#
#       Dependency : corpus needs to be available
#
#       Parameters for main() method:
#                    1 => corpus
#                    2 => keywordFilePath
#                    3 => topn
#
# ========================================================================================================

from contextlib import redirect_stdout
import os
#import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk

en_stop = set(nltk.corpus.stopwords.words('english'))
#from scipy.sparse import coo_matrix

with redirect_stdout(open(os.devnull, "w")):
    nltk.download('stopwords')


#Function for sorting tf_idf in descending order
def sort_coo(coo_mat):
    tuples = zip(coo_mat.col, coo_mat.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


def main(corpus,keywordFilePath,topn=None):
    
    if topn == None:
        topn=10
    else:
        topn=int(topn)
        
    cv=CountVectorizer(max_df=0.8,max_features=10000,stop_words=en_stop,ngram_range=(1,2))
    X=cv.fit_transform(corpus)
     
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(X)
    # get feature names
    feature_names=cv.get_feature_names()
     
    # fetch document for which keywords needs to be extracted
    
    keywordsList=set()
    
    
    for i in range(len(corpus)):
        doc=corpus[i]
         
        #generate tf-idf for the given document
        tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
        #sort the tf-idf vectors by descending order of scores
        sorted_items=sort_coo(tf_idf_vector.tocoo())
        #extract only the top n; n here is 10
        keywords=extract_topn_from_vector(feature_names,sorted_items,topn)
        for keyword in keywords:
            keywordsList.add(keyword)
            
            
    with open(keywordFilePath,'w') as fout:
        for line in keywordsList:
            fout.write(line)
            fout.write('\n')