#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =======================================================================================================
# 
#       Name : addnode.py
#       Description: To add new nodes in the existing Taxonomy
#       Created by : Mili Biswas
#       Created on: 19.04.2020
#
#       Dependency : Software needs to be installed first
#                    Existing taxonomy needs to be available.
#
#       Parameters:
#               1 => corpus (new/incremental)
#               2 => Taxonomy (JSON)
#               3 => Type of similarity measure (optional. e.g. Jaccard, Cosine, Eucleadian etc)
# ========================================================================================================

from gensim.models import FastText
import numpy as np
from numpy import dot
from numpy.linalg import norm
import json
import sys

class genvec(object):
    '''
       This class is for generating the vectors by retraining the new corpus using transfer learning
    '''
    
    def __init__(self,savedModel):
        self.savedModel=savedModel
        self.model=None
        
    def __loadModel(self,):
        print('[Info]: Loading the saved model')
        self.model=FastText.load(self.savedModel)
        
    def __updateVocab(self,corpus):
        print('[Info]: Updating the vocabulary with new data')
        self.model.build_vocab(corpus,update=True) 
        
    def __train(self,corpus):
        print('[Info]: Retrain the model with new data')
        total_examples = self.model.corpus_count 
        self.model.train(sentences=corpus, total_examples=total_examples, size=60,epochs=10)
        
    def retrain(self,corpus):
        try:
            self.__loadModel()
            self.__updateVocab(corpus)
            self.__train(corpus)
            print('[Info]: Model retraining is successful')
        except Exception as err:
            print('[Error]: '+ str(err))
            sys.exit(1)
            
    def getWordVector(self,wordList):
        output=[]
        for w in wordList:
            try:
                tempList=[]
                tempList.append(w)
                tempList=tempList+list(self.model.wv[w])
                output.append(tempList)
            except Exception as err:
                print('[Error]:'+str(err))
                continue
        return output
        
class taxonomy(object):
    def __init__(self,inputTaxonomy=None,inputSimilarityType='cosine'):
        self.inputTaxonomy=inputTaxonomy
        self.inputSimilarityType=inputSimilarityType.lower()
        self.taxonomyTree=None
        self.embeddingList=None
        
        try:
            if self.inputTaxonomy==None:
                raise Exception
        except Exception as e:
            print('[Error]: '+ str(e))
            
    def __setTree(self,inputJsonFile):
        '''
           Saving the tree in a variable
        '''
        try:
            with open(inputJsonFile) as jsonFile:
                jsonData=json.load(jsonFile)
            self.taxonomyTree=jsonData
        except Exception as e:
            print('[Error]: '+ str(e))
            sys.exit(1)
            
    def __getWordVectors(self,taxonomyTree):
        
        '''
           This function extracts all the vectors of nodes from tree
        '''
        
        nodeVectorsList=[]
        
        for child in taxonomyTree['children']:
            if child:
                retVal=self.__getWordVectors(child)
                nodeVectorsList = nodeVectorsList+ retVal
            
        nodeVectorsList.append(taxonomyTree['center'])
        return nodeVectorsList
        
    def __getSimilarityScore(self,vList,vec):
        scoreList=[]
        for v in vList:
            if v:
                center=np.array(v)
                scoreVal = dot(center, vec)/(norm(center)*norm(vec))  
                scoreList.append(scoreVal)
            else:
                scoreList.append(2)
        return scoreList
    
    def __getMaxSimilarityVal(self,vList):
        val=min(vList)
        return val,vList.index(val)
    
    def __insertNode(self,taxonomyTree,pNode,nodeData):
        
        if taxonomyTree['center'] == pNode:
            taxonomyTree['children'].append(nodeData)
            return
        else:
            for child in taxonomyTree['children']:
                self.__insertNode(child,pNode,nodeData)
            return
                    
    def __addNode(self,inputTree,vectorList):
        
        # Get all the vectors of every nodes in the tree
        treeVecs=self.__getWordVectors(inputTree)
        
        # Loop over the list of keywords vectors to be checked and added in the tree
        
        for elem in vectorList:
            
            word=elem[0]           # The new keyword
            
            vec=np.array(elem[1:],dtype=float) # The new keyword's vector
            
            nodeData={}            # Dictionary for holding new keywords data while added into the tree
            
            scoreList=self.__getSimilarityScore(treeVecs,vec)  # Similarity score calculation among each keyword vector and nodes in the tree
            
            maxScore,pos=self.__getMaxSimilarityVal(scoreList) # Identifying most similar node w.r.t the new keyword 
            
            # Filling up the node data dictionary before adding the new keyword into the tree
            
            nodeData['id']=word
            nodeData['name']=word
            nodeData['center']=list(vec)
            nodeData['data']={'type':'concept','depth':99}
            nodeData['children']=[]
            
            # This is for adding the new keyword in the desired position in the tree
            
            self.__insertNode(self.taxonomyTree,treeVecs[pos],nodeData)
            
    def __createJavaScriptFile(self,path=None):
        try:
            if self.taxonomyTree==None:
                raise Exception
            else:
                with open(path,'w') as fout:
                    json.dump(self.taxonomyTree,fout)
                print('[Info]: Json file created at :',path)
        except Exception as err:
            print('[Error]: {}'.format(str(err)))
            sys.exit(1)
    
    def process(self,vectorList,jsonFile):
        self.__setTree(jsonFile)
        self.__addNode(self.taxonomyTree,vectorList)
        self.__createJavaScriptFile('./src/data/output/enhanced_taxonomy.json')
        return None