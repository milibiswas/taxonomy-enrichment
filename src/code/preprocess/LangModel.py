#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:01:46 2020

@author: milibiswas

"""
from collections import Counter
from nltk import FreqDist
import itertools as it
import spacy 
import pandas as pd 
pd.set_option('display.max_colwidth', 200)
nlp=spacy.load("en_core_web_sm")
import pandas as pd
from taxonomy.data.preprocess import Data

class LanguageModel(Data):
    def __init__(self,dataObject):
        super().__init__(self)
        self.processedCorpusOnTag=None
        self.freqDist=None
        self.data=dataObject
        self.contextData=None
    
    def __subtree_matcher(self,doc):
        lst=[]
        for tok in doc:
            if tok.pos_=="NOUN" or tok.pos_=='PROPN' or tok.text.find('_') != -1:
                lst.append(tok.text)
        return lst
    
    def __buildCorpusOnTag(self,):
        corpus=[]
        for cnt,line in enumerate(self.data.outputProcessedCorpus):
            doc=nlp(line)
            tmp=([super(LanguageModel,self).removeStopWords(w) for w in self.__subtree_matcher(doc) if super(LanguageModel,self).removeStopWords(w)])
            if tmp:
                corpus.append(tmp)
            
        self.processedCorpusOnTag=corpus
        
    def __context_window(self,sequence, n):
        '''
            Context window
        '''
        data={}
        for i in range(len(sequence)):
            if i==0:
                data[sequence[i]]=[ j for j in sequence[i+1:n+1]]
            elif i<n:
                data[sequence[i]]=[ j for j in sequence[0:i]] + [ j for j in sequence[i+1:n+i+1]]
            else:
                data[sequence[i]]=[ j for j in sequence[i-n:i]] + [ j for j in sequence[i+1:n+i+1]]
        return data
        
# =============================================================================
#     def __compute_ngrams_with_window(self,sequence, n):
#         return zip(*[sequence[index:] for index in range(n)])
#     
#     def __compute_ngrams_without_window(self,sequence, n):
#         ngramList=[]
#         try:
#             for e in list(it.combinations(sequence,n)):
#                 ngramList.append(e)
#             return ngramList
#         except Exception as err:
#             print(err)
# =============================================================================
        
            
# =============================================================================
#     def __frequencyDist(self,ngramsList):
#         
#         tmp=[]
#         for lst in ngramsList:
#             for tup in lst:
#                 tmp.append('#'.join(sorted(tup)))
# 
#         fd = FreqDist(tmp)
#         ferquencyDist={}
#         
#         #tot_number_ngrams=len(list(fd.keys()))
#         for key,val in fd.items():
#             if key not in ferquencyDist:
#                 ferquencyDist[key]=[]
#             ferquencyDist[key]=[val,tuple(key.split('#'))]            
#         self.freqDist=ferquencyDist
# =============================================================================
        
    def __most_common_context(self,ngramDict,mostCommonContext):
        contextList=[]
        for key,val in ngramDict.items():
            tmp=[]
            for context,_ in val.most_common(mostCommonContext):
                tmp.append(list((key,context)))
            if tmp:
                contextList += tmp
        return contextList
        
        
    def callLanguageModel(self,windowSize=5,mostCommonContext=10):
        self.__buildCorpusOnTag()
        
        ngramDict={}
        
        for lst in self.processedCorpusOnTag:
            dataDict=self.__context_window(lst,windowSize)
            for key in dataDict:
                if key not in ngramDict:
                    ngramDict[key]=Counter()
                context=dataDict[key]
                tmp=list(ngramDict[key].elements())+context
                ngramDict[key]=Counter(tmp)
                
        self.contextData=ngramDict
            
        return self.__most_common_context(ngramDict,mostCommonContext)