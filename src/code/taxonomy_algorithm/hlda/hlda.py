#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#     Name: hlda.py
#     Descirption: This python script is to generate taxonomy using 
#                  hierarchical latent dirichlet allocation (hlda)
#     Created by: Mili Biswas (Department of Informatics, UNIFR, Switzerland)
#     Creation Date: 23-Feb-2020
#     Parameters:
#                 corpus=sys.argv[1]
#                 keywords=sys.argv[2]
#                 depth=sys.argv[3]
#                 jsonFilePath=sys.argv[4]
# =============================================================================

import tomotopy as tp
import json
import sys

def getTopicWords(topic_id,n_top_words=10):
  '''
      This is a wrapper function on get_topic_words() from tomotopy
      which returns only top n (n is a variable) words.
  '''
  list_of_words=[ w for w,_ in mdl.get_topic_words(topic_id,top_n=n_top_words) if w in list_of_keywords]
  
  return list_of_words
  
  
def topicTaxonomy(model,rootTopicId=0):
  '''
       This funtion runs recursively to build the topic taxonomy 
       from the output of tomotopy HLDA model
  '''
  wordList=getTopicWords(rootTopicId,1000)
    
  if mdl.is_live_topic(rootTopicId) and wordList:
    myDict={}
    myDict['topic_id']=rootTopicId
    myDict['name']=wordList
    myDict['children']=[]

    if len(model.children_topics(rootTopicId))>0:
      for child in model.children_topics(rootTopicId):
        
        ret=topicTaxonomy(model,child)
        if ret:
          myDict['children'].append(ret)
      return myDict

    else:
      return myDict
    
  else:
    return None

if __name__ == '__main__':
    
    corpus=sys.argv[1]
    keywords=sys.argv[2]
    depthVal=sys.argv[3]
    jsonFilePath=sys.argv[4]
    
    # =============================================================================
    #  model declaration
    # =============================================================================
    
    mdl = tp.HLDAModel(depth=int(depthVal))
    
    # =============================================================================
    #  data preparation and adding to the model
    # =============================================================================
    
    for line in open(corpus):
        mdl.add_doc(line.strip().split())
    
    # =============================================================================
    # training the model
    # =============================================================================
    
    for i in range(0, 100, 50):  # TODO : The values can be parametrized is needed
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    
    # =============================================================================
    #  List of keywords
    # =============================================================================
    
    list_of_keywords=[]
    with open(keywords,'r') as fin:
        for line in fin:
            list_of_keywords.append(line.strip('\n'))


    # =============================================================================
    #  calling topicTaxonomy() function    
    # =============================================================================
    
    dataDict=topicTaxonomy(model=mdl,rootTopicId=0)
    
    # =============================================================================
    #  Generating the JSON file
    # =============================================================================
    
    with open(jsonFilePath,'w') as fout:
        json.dump(dataDict,fout)
