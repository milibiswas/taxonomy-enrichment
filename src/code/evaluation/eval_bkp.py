#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:31:21 2020

@author: milibiswas
"""

import sys
#import re
import os
import json
from nltk import FreqDist

class Metric(object):
    def __init__(self,):

        self.pathList=[]
        self.dataDict=None
        
    def getJason(self,path='./output/json_fashion_data.js'):
        try:
            if self.dataDict==None:
                raise Exception
            else:
                with open(path,'w') as fout:
                    json.dump(self.dataDict,fout)
                print('Jason file created at :',path)
        except Exception as err:
            print('Error occurred')
            print(str(err))
            sys.exit(1)
            
    def dumpPathList(self,pathListFile='./output/pathlist.txt'):
        try:
            if pathListFile==None:
                raise Exception
            else:
                with open(pathListFile,'w') as fout:
                    for line in self.pathList:
                        fout.write('-->'.join(line))
                        fout.write('\n')
                print('pathlist data file created at :',pathListFile)
        except Exception as err:
            print('Error occurred')
            print(str(err))
            sys.exit(1)
        
        
    def __rootToLeafPath(self,inputTree,nameStr=''):
        '''
            This function will produce a list of lists
            where each individual inner list is a distinct
            path from root to leaves
        '''
        if nameStr=='':
            nameStr=inputTree['name']
        else:
            nameStr=nameStr+'#'+inputTree['name']
            
        if len(inputTree['children'])>0:
            for child in inputTree['children']:
                self.__rootToLeafPath(child,nameStr)
        else:
            self.pathList.append(nameStr.split('#'))
        
        return None
    
    def __nodeRenameAlgorithm(self,seedFile):
        '''
            This function implements node rename algorithm which the json builder
            will use subsequently to generate the Taxonomy & Publish in JavaScript
            HyperTree.
        
        '''
        #print("========================= Starting nodeRenameAlgorithm() function ==========================")
        wordList=[]
        with open(seedFile,'r') as fin:
            for line in fin:
                keyWord=line.strip().strip('\n')
                if '_' in keyWord:
                    temp=keyWord.split('_')
                    wordList += temp
                else:
                    wordList.append(keyWord)
                    
        # Frequency Distribution of words
        
        dist=FreqDist(wordList)
        
        nodeNamesList=[]
        for w,val in dist.most_common(2):
            if val>1:
                nodeNamesList.append(w)
            else:
                nodeNamesList.append('other')
            
        return ' & '.join(list(set(nodeNamesList)))
    
    def jsonBuilder(self,inputDir='/Users/milibiswas/Desktop/Master-Thesis/run_taxogen_9L_HCLUS/data/dblp/our-l3-0.15',parent='Fashion',level=1):
    
        dataDict={}
        if parent=='Fashion':
            pass
        else:
            parent=self.__nodeRenameAlgorithm(os.path.join(inputDir,'seed_keywords.txt'))
        
        '''
            1. Read parent/root directory
            2. Locate hierarchy File in the directory
            3. Read Hierarchy file
            4. Populate general fields from the directory + Hierarchy file
            5. If there are children from hierarchy, call the self function for each child.
            6. Add each childs return dict into a list (children list maintained in parent)
    
        '''
    
        
        if os.path.exists(os.path.join(inputDir,'hierarchy.txt')):
            children=[]
            #parent=''
                
            with open(os.path.join(inputDir,'hierarchy.txt'),'r') as fin:
                for line in fin:
                    child,parent1=line.strip().split()
                    children.append(child)
                    #parent=parent
            dataDict['id']=parent        
            dataDict['name']=parent
            dataDict['children']=[]
            
            # Other than root level, we need data element
            if level>1:
                dataDict['data']={"type":"concept","depth":level}
                
            else:
                dataDict['queries']="false"
                dataDict['description']="Root Level"
                
            for child in children:
                ret=self.jsonBuilder(os.path.join(inputDir,child),child,level+1)
                dataDict['children'].append(ret)                
            return dataDict
        else:
            try:
                dataDict['id']=parent        
                dataDict['name']=parent
                dataDict["data"]={"type":"concept","depth":level}
                leaf_nodes=[]
                with open(os.path.join(inputDir,'seed_keywords.txt'),'r') as fin:
                    for line in fin:
                        leaf_dict={}
                        leaf_dict["id"]=line.strip()
                        leaf_dict["data"]={"type":"concept","depth":level}
                        leaf_dict["name"]=line.strip()
                        leaf_dict["children"]=[]
                        leaf_nodes.append(leaf_dict)
                dataDict['children']=leaf_nodes
                return dataDict
            except Exception as err:
                print('[Error]:',str(err))
                return
            
# =============================================================================
#      Relational Accuracy - Calculation   
# =============================================================================
    @staticmethod        
    def measureRelationalAccuracy(actualResult,groundTruth):
        '''
            This function resturns the relational accuracy mesurements.
            
            Parameters:
                        1. Ground Truth  --> This will come from File
                        2. Actual Value  --> This is a python list if lists
            Note:
                  Each file should be with proper format!
        
        '''
    
        gtList= []
        with open(groundTruth,'r') as fin:
            for cnt,line in enumerate(fin):
                if line:
# =============================================================================
#                     x=re.sub(" & ","%%",str(line.strip("\n")))
#                     y=re.sub(" ","-->",x)
#                     z=re.sub("%%"," & ",y)
# =============================================================================
                    gtList.append(line.split('-->'))
        ptList= []
        with open(actualResult,'r') as fin:
            for cnt,line in enumerate(fin):
                if line:
# =============================================================================
#                     x=re.sub(" & ","%%",str(line.strip("\n")))
#                     y=re.sub(" ","-->",x)
#                     z=re.sub("%%"," & ",y)
# =============================================================================
                    ptList.append(line.split('-->'))
        
        
        print(len(gtList),len(ptList))
        return ('RA Score : ',(len(gtList)/len(ptList)))
    
    
# =============================================================================
#      NMI - Calculation   
# =============================================================================
    @staticmethod 
    def measureNMI():
        pass
    
# =============================================================================
#      Performance of Clustering - Davies Bouldin Index
# =============================================================================
        
    @staticmethod
    def measureDBI():
        pass
    
# =============================================================================
#      NMI - Calculation   
# =============================================================================
    
    
    def evaluate(self,path,parent='Fashion',level=1):
        try:
            self.dataDict=self.jsonBuilder(path,parent,level)
            self.__rootToLeafPath(self.dataDict)
            
            print('Path list is ready')
        except Exception as err:
            print('Error in evaluation - check the errors for details')
            print(str(err))
            sys.exit(1)