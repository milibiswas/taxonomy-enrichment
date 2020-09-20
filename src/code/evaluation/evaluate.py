#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =======================================================================================================
# 
#       Name : evaluate.py
#       Description: To run the evaluation
#       Created by : Mili Biswas
#       Created on: 27.02.2020
#
#       Dependency : Software needs to be installed first. This script should be called by run.py.
#                    However, independent call is also possible.
#
#       Parameters:
#
# ========================================================================================================

import numpy as np
import pandas as pd
import random
import string
import statistics
import json
import operator
import os
import random

from sklearn.metrics.cluster import v_measure_score,adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics import f1_score
from sklearn.metrics import davies_bouldin_score

class Evaluation(object):
    def __init__(self,buildTaxonomyFile=None,groundTruthFile=None):
        self.pathList=[]
        self.dataDict=None
        self.groundTruthFile=groundTruthFile
        self.buildTaxonomyFile=buildTaxonomyFile
        self.groundTruth=None
        self.keywords2label={}
        
# =============================================================================
#     def __json2pythonDataStructure(self,):
#         try:
#             with open(self.actualJasonData) as f:
#               data = json.load(f)
#             self.dataDict=data
#             print(self.dataDict)
#             return True
#             
#         except Exception as err:
#             print('[Error]ok2:',str(err))
#             return False
# =============================================================================
        
    def __fileExist(self,file):
        if os.path.exists(file):
            return True
        return False
    
    def __processTaxonomyFile(self,):
        if not self.buildTaxonomyFile:
            print('[Warning]: The taxonomy file is not provided')
        else:
            if self.__fileExist(self.buildTaxonomyFile):
                jsonFile=open(self.buildTaxonomyFile)
                self.dataDict=json.load(jsonFile)
            else:
                print("[Error]: Taxonomy file {} not found or can't be accessed".format(self.buildTaxonomyFile))
        
    
    def __processGroundTruthFile(self,):
        if not self.groundTruthFile:
            print('[Warning]: Ground Truth file is not provided')
        else:
            if self.__fileExist(self.groundTruthFile):
                gtDataList=[]
                with open(self.groundTruthFile,'r') as fin:
                    for line in fin:
                        recList=line.strip('\n').split(',')
                        gtDataList.append(recList[:-1])
                        self.keywords2label[recList[-2]] = recList[-1]
                self.groundTruth=gtDataList
            else:
                print("[Error]: Ground Truth file {} not found or can't be accessed".format(self.groundTruthFile))
                
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

         
    def process(self,):
        try:
            self.__processTaxonomyFile()
            self.__processGroundTruthFile()
            self.__rootToLeafPath(self.dataDict)
            #print(self.pathList)
        except Exception as err:
            print('[Error]: process() method in Evaluation class instance:')
            print('[Error]: ',str(err))


    #============================================================================================
    #   Static methods : These are for calculating NMI & F1 scores
    #============================================================================================
    
    @staticmethod
    def randomNumberGenarator(low,high,no_of_sample):
        return random.sample(range(low, high), no_of_sample)
        
    
    @staticmethod 
    def randomString(stringLength=5):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))
    
    @staticmethod 
    def getMaxLength(ll):
        '''
           Algorithm that gives maximum length of the  list
           from a list of lists
        '''
        maxVal=0
        for l in ll:
            tmp=len(l)
            if tmp >= maxVal:
                maxVal=tmp
            else:
                pass
        
        return maxVal
    
    
    @staticmethod 
    def lookupLabel(keyword,keyword2label):
        return keyword2label[keyword]
    
    @staticmethod 
    def harmonizedActualResults(nmi_ar,nmi_gt,keyword2label):
        
        '''
            Algorithm to make actual results & ground truth harmonized
            to calculate the NMI score. This is needed to successfully 
            calculate NMI & F1 score. We have to make the sample size 
            same for actual results as well as ground truth data.
        '''
        
        # First make the set of the actual result data & ground truth data
        nmi_ar_set=set(nmi_ar)
        nmi_gt_set=set(nmi_gt)
        
        #This is common dataset between actual result and ground truth data
        set_common=nmi_ar_set.intersection(nmi_gt_set)
        
        #Here difference in data between ground truth & actual result (both ways in set)
        l_diff = nmi_ar_set - nmi_gt_set
        r_diff = nmi_gt_set - nmi_ar_set
        
        #Empty lists inialization for holding ground truth labels & actual result labels
        ar_list=[]
        gt_list=[]
        
        randList=Evaluation.randomNumberGenarator(1999999,2999999,len(l_diff))
        
        #Building the ground truth labels & actual result labels
        for i in set_common:
            ar_list.append(Evaluation.lookupLabel(i,keyword2label))
            gt_list.append(Evaluation.lookupLabel(i,keyword2label))
        for c,i in enumerate(l_diff):
            #ar_list.append(Evaluation.lookupLabel(i,keyword2label))
            ar_list.append(randList[c])
            gt_list.append(999999)  # A big value
            
# =============================================================================
#         for i in r_diff:
#             gt_list.append(Evaluation.lookupLabel(i,keyword2label))
#             ar_list.append(888888)  # A big value
# =============================================================================
    
        return (ar_list,gt_list)
    
    
    @staticmethod     
    def measureNMI(groundTruth, actualResult,keywords2label):
        
            '''
                Algorithm to calculate the NMI score using ground truth & actual result
                This also calculates the F1 scores using the same dataset
            
            '''
            
            f1_score={}
            nmi_score={}
            gtDataFrame=pd.DataFrame(groundTruth)
            arDataFrame=pd.DataFrame(actualResult)

            
            # This is giving the maximum lenth of the list from list of lists of actual result
            upperLimit=Evaluation.getMaxLength(actualResult)
            
            try:
                for level in range(0,upperLimit):
                    
                    for i in list(arDataFrame[level].unique()):    # e.g. Fashion in level 0 etc.
                        # Here get the category value from actual result tree
                        nmi_ar=[]
                        scoreIndex={}
                        keepGTList=[]
                        
                        for l in actualResult:
                            if len(l)-1>level:
                                if l[level]==i:
                                    nmi_ar.append(l[-1])   # Keeping the labels for each cluster
                        
                        # Here for each value of actual category, get all possible categories from same level in ground truth
                        cnt=0
                        
                        # Here for each category of a level from actual result, we find all the categories of the same level
                        # from the ground truth. Then we pick the maximum matching cluster from ground truth to calculate the NMI
                        
                        for j in list(gtDataFrame[level].unique()):
                            nmi_gt=[]
                            for l in groundTruth:
                                if len(l)-1>level:
                                    if l[level]==j:
                                        nmi_gt.append(l[-1])
                                        
                            keepGTList.append(nmi_gt)
                            arSet=set(nmi_ar)
                            gtSet=set(nmi_gt)
                            commonElementsNumber=arSet.intersection(gtSet)
                            if cnt not in scoreIndex:
                                scoreIndex[cnt]=0
                            scoreIndex[cnt]=len(commonElementsNumber)    
                            cnt = cnt + 1
                            
                        if nmi_ar:
                            # This is needed to get the index of maximum matching cluster
                            max_key=max(scoreIndex.items(), key=operator.itemgetter(1))[0]
                            
                            # This is to get the maximum matched list
                            max_nmi_gt=keepGTList[max_key]
                            
                            # This is to compare the actual result set and ground truth set
                            ar,gt = Evaluation.harmonizedActualResults(nmi_ar,max_nmi_gt,keywords2label)
                            
                            # Here calculating the NMI score
                            score=v_measure_score(ar,gt)

                            # Here calculating the F1 score
                            f1score=Evaluation.measureF1Score(gt,ar)
                            
                            # Here below, we are keeping the NMI scores in a dictionary for averaging the values later.
                            
                            if level not in nmi_score:
                                nmi_score[level]=[]
                            if nmi_ar:
                                nmi_score[level].append(score)
                                
                            # Here below, we are keeping the F1 scores in a dictionary for averaging the values later.
                                
                            if level not in f1_score:
                                f1_score[level]=[]
                            if nmi_ar:
                                f1_score[level].append(f1score)
                            
                #This part is calculating the levelwise NMI by averaging the values of each level
                
                finalNMI={}
                for key,val in nmi_score.items():
                    finalNMI[key]=statistics.mean(val)
                
                
                #This part is calculating the F1 scores of the hierarchy by averaging the values of all level's data
                
                # Define an empty dictionary to hold F1 score.
                finalF1Score={}
            
                # This lists are to hold micro, macro & weight F1 scores
                fmicro=[]
                fmacro=[]

                # Here iterating through the earlier filled dictionary with F1 scores from all levels.
                for level,val in f1_score.items():
                    fmic=[]
                    fmac=[]
                    
                    for d in val:
                        fmic.append(d['micro'])
                        fmac.append(d['macro'])
                        
                    # Here calculating and keeping the average values of each level    
                    fmicro.append(statistics.mean(fmic))
                    fmacro.append(statistics.mean(fmac))
                    
                # Here calculating the average values from all levels
                if fmicro:
                    finalF1Score['micro']=statistics.mean(fmicro)
                    #print(finalF1Score['micro'])
                    
                if fmacro:
                    finalF1Score['macro']=statistics.mean(fmacro)
                    #print(finalF1Score['macro'])
                    
                
                # Returning the finally calculated average values
                return (finalNMI,finalF1Score)
            except Exception as err:
                print(err)
        
    @staticmethod
    def measureDBI(inputData,inputLabel):
        return davies_bouldin_score(inputData,inputLabel)
    
    @staticmethod
    def measureF1Score(yTrue=None,yPred=None):
        '''
             This function calculates the F1 scores
        '''
        
        f1Macro=f1_score(yTrue, yPred, average='macro',labels=np.unique(yPred))
        f1Micro=f1_score(yTrue, yPred, average='micro',labels=np.unique(yPred))
        #f1Weighted=f1_score(yTrue, yPred, average='weighted',labels=np.unique(yPred))
        
        return {'macro':f1Macro,'micro':f1Micro}    