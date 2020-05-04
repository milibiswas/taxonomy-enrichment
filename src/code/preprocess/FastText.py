#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 12:52:48 2020

@author: milibiswas
"""

from gensim.models import FastText as ft_model
import sys

#---------------------------------
# Model class - skipgram
#---------------------------------
        
        
class FastText(object):
    '''
            Fasttext -> from gensim
    '''
    def __init__(self,corpus=None,size=60,window=5,min_count=5,sample=1e-4,sg=1,iter=5,seed=0):
        self.corpus=corpus
        self.size=size
        self.window=window
        self.min_count=min_count
        self.sample=sample
        self.sg=sg
        self.iter=iter
        self.seed=seed
        self.model=None
        
    def save(self,path):
        self.model.save(path)
        
    def train(self,):
        # Hyper parameters
        #negative_sampling=5
        model=ft_model(
                           sentences=self.corpus,
                           size=self.size,
                           window=self.window,
                           min_count=self.min_count,
                           sample=self.sample,
                           sg=self.sg,
                           iter=self.iter,
                           seed=self.seed
                           )
        
        self.model=model
        print('[Info]: Model is trained')
    
    def __getVocabulary(self,keywordList):
        inpVoc=list(set(keywordList))
        voc=[w for w in inpVoc if w in self.model.wv.vocab]
        return voc
               
    def getWordEmbeddings(self,keywordList=None):
        filtered_voc=self.__getVocabulary(keywordList)
        word_vectors=[]
        vocab=[]
        for w in filtered_voc:
            try:
                word_vectors.append(self.model[w])
                vocab.append(w)
            except Exception:
                #print(str(err))
                #print(w)
                continue
        return (vocab,word_vectors)
    
    def saveWordEmbeddings(self,path='./output/embeddings.txt',keywordList=None):
        try:
            voc,vec=self.getWordEmbeddings(set(keywordList))
            dimension=len(vec[0])     # This will give the dimension of each word vector
            no_of_words=len(voc)      # This gives the number of words in vocabulary
            
            with open(path,'w') as fout:
                fout.write(str(no_of_words)+' '+str(dimension)+'\n')
                for i,w in enumerate(voc):
                    fout.write(w)
                    for feature in vec[i]:
                       fout.write(' '+str(feature)) 
                    fout.write('\n')
        except Exception as err:
            print('Error occurred in saveWordEmbedding() method, check logs for details analysis')
            print(str(err))
            sys.exit(1)
            
    def savekeywords(self,path='./output/keywords.txt',keywordList=None):
        try:
            voc,vec=self.getWordEmbeddings(set(keywordList))
            cnt=0
            
            with open(path,'w') as fout:
                for i,w in enumerate(set(voc)):
                    fout.write(w)
                    fout.write('\n')
                    cnt+=1
            print('[Info]: Keywords save : total {}'.format(cnt))
            
        except Exception as err:
            print('[Error]: Error occurred in savekeywords() method, check logs for details analysis')
            print(str(err))
            sys.exit(1)
            
    def saveCorpus(self,path='./output/papers.txt'):
        try:
            
            with open(path,'w') as fout:
                for blog in self.corpus:
                    fout.write(' '.join(blog))
                    fout.write('\n')
            print('[Info]: Corpus saved at : ',path)
            
        except Exception as err:
            print('Error occurred in saveCorpus() method, check logs for details analysis')
            print(str(err))
            sys.exit(1)
        
    
if __name__ == '__main__':
    obj=FastText()
    obj.train()