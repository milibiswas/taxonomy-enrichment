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

from gensim.models import FastText,Word2Vec,KeyedVectors
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance
import json
import sys
import os

class genvec(object):
    '''
       This class is for generating the vectors by retraining the new corpus using transfer learning
    '''
    
    def __init__(self,savedModelPath):
        self.savedModelPath=savedModelPath
        self.model=None
        
    def __getVocab(self,):
        
        print('[Info]: Getting vocabulary from the saved models')
        
        dirName=self.savedModelPath
        vocabList=[]
        for file in os.listdir(dirName):
            model=KeyedVectors.load_word2vec_format(os.path.join(dirName,file), binary=False)
            vocabList.extend(list(model.vocab.keys()))
        return vocabList
        
    def __loadModel(self,inModel):
        print('[Info]: Loading the saved model')
        dirName=self.savedModelPath
        for file in os.listdir(dirName):
            #if os.path.basename(os.path.join(dirName,file))=='sport_pant':
                model=KeyedVectors.load_word2vec_format(os.path.join(dirName,file), binary=False)
                inModel.build_vocab([list(model.vocab.keys())], update=True)
                inModel.intersect_word2vec_format(os.path.join(dirName,file), binary=False, lockf=0.0)
        return inModel
        
        
        
    def __updateVocab(self,corpus):
        print('[Info]: Updating the vocabulary with new data')
        self.model.build_vocab(corpus,update=True) 
        
    def __train(self,corpus):
        print('[Info]: Retrain the model with new data')
        
        model_new = Word2Vec(size=60, min_count=1)   # Initialized the w2v model with new data
        model_new.build_vocab(corpus)                # Initialized model with vocabulary from new data
        
        total_examples = model_new.corpus_count      # Number of examples
        
        retModel=self.__loadModel(model_new)         # Loading pretrained model
        
        retModel.train(
                        sentences=corpus, 
                        total_examples=total_examples,
                        epochs=200)        # This is retraining part        
        self.model=retModel
        
    def retrain(self,corpus):
        try:
            #self.__loadModel()
            #self.__updateVocab(corpus)
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
           This function extracts all the vectors of nodes from tree except leaf nodes
        '''
        
        nodeVectorsList=[]
        
        l_children=len(taxonomyTree['children'])  # This length is needed to filter out leaf nodes
        
        if l_children>0:
            for child in taxonomyTree['children']:
                retVal=self.__getWordVectors(child)
                if retVal:
                    nodeVectorsList = nodeVectorsList+ retVal
        else:
            return None
        if len(nodeVectorsList)==0:    
            nodeVectorsList.append(taxonomyTree['center'])
            return nodeVectorsList
        else:
            return nodeVectorsList
        
    def __getSimilarityScore(self,vList,vec):
        '''
            This function calculates the similarity score (cosine)
        '''
        scoreList=[]
        for v in vList:
            if v:
                center=np.array(v)
                #scoreVal = np.inner(center, vec) / (norm(center) * norm(vec)) 
                #scoreVal=1-distance.cosine(center, vec)   # Cosine Similarity
                scoreVal = np.linalg.norm(center-vec)      # Eucleadian distance
                scoreList.append(scoreVal)
            else:
                scoreList.append(900000)  # This is for those nodes which are not being considered
        return scoreList
    
    def __getMaxSimilarityVal(self,vList):
        '''
            This function calculates the maximum cosine similarity scores or minimum Eucleadian distance
        '''
        
        val=min(vList)
        return val,vList.index(val)
    
    def __insertNode(self,taxonomyTree,pNode,nodeData):
        '''
           This function inserts the node under the provided parent node
        '''
        
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
            
            #print(scoreList)
            
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
                with open(os.path.join(path,'enhanced_taxonomy.json'),'w') as fout:
                    json.dump(self.taxonomyTree,fout)
                print('[Info]: Json file created at :',path+'/enhanced_taxonomy.json')
                
                with open(os.path.join(path,'hypertree/Visualisation/json_data.js'),'w') as fout:
                    fout.write('function json_data() {')
                    fout.write('\n')
                    fout.write('var x = ')
                    json.dump(self.taxonomyTree,fout)
                    fout.write(';')
                    fout.write('return x; }')
                    
        except Exception as err:
            print('[Error]: {}'.format(str(err)))
            sys.exit(1)
    
    def process(self,vectorList,jsonFile):
        self.__setTree(jsonFile)
        self.__addNode(self.taxonomyTree,vectorList)
        self.__createJavaScriptFile('./src/data/output')
        return None