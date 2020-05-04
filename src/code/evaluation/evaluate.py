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

from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import f1_score
from sklearn.metrics import davies_bouldin_score

class Evaluation(object):
    def __init__(self,gtList,dataDict):
        self.gtList=gtList
        self.pathList=[]
        self.dataDict=dataDict
        
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
            self.__rootToLeafPath(self.dataDict)
        except Exception as err:
            print('[Error]: process() method in Evaluation class instance:')
            print('[Error]: ',str(err))

        
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
            to calculate the NMI score
        '''
        
        nmi_ar_set=set(nmi_ar)
        nmi_gt_set=set(nmi_gt)
        set_common=nmi_ar_set.intersection(nmi_gt_set)
        l_diff = nmi_ar_set - nmi_gt_set
        r_diff = nmi_gt_set - nmi_ar_set
        ar_list=[]
        gt_list=[]
        
        for i in set_common:
            ar_list.append(Evaluation.lookupLabel(i,keyword2label))
            gt_list.append(Evaluation.lookupLabel(i,keyword2label))
        for i in l_diff:
            ar_list.append(Evaluation.lookupLabel(i,keyword2label))
            gt_list.append(999999)  # A big value
        for i in r_diff:
            gt_list.append(Evaluation.lookupLabel(i,keyword2label))
            ar_list.append(888888)  # A big value
    
        return (ar_list,gt_list)
    
    
    @staticmethod     
    def measureNMI(groundTruth, actualResult,keywords2label):
        
            '''
                Algorithm to calculate the NMI score using ground truth & actual result
            
            '''
            f1_score={}
            nmi_score={}
            gtDataFrame=pd.DataFrame(groundTruth)
            arDataFrame=pd.DataFrame(actualResult)
            
            # This is giving the maximum lenth of the list from list of lists
            upperLimit=Evaluation.getMaxLength(actualResult)   
            
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
                        max_key=max(scoreIndex.items(), key=operator.itemgetter(1))[0]
                        max_nmi_gt=keepGTList[max_key]
                        ar,gt = Evaluation.harmonizedActualResults(nmi_ar,max_nmi_gt,keywords2label)
                        
                        score=v_measure_score(ar,gt)
                        f1score=Evaluation.measureF1Score(gt,ar)
                        if level not in nmi_score:
                            nmi_score[level]=[]
                        if nmi_ar:
                            nmi_score[level].append(score)
                            
                        if level not in f1_score:
                            f1_score[level]=[]
                        if nmi_ar:
                            f1_score[level].append(f1score)
                        
                        
            finalNMI={}
            for key,val in nmi_score.items():
                finalNMI[key]=statistics.mean(val)
            
            finalF1Score={}
            fmicro=[]
            fmacro=[]
            fweight=[]
            
            for key,val in f1_score.items():
                fmic=[]
                fmac=[]
                fwgh=[]
                for d in val:
                    fmic.append(d['micro'])
                    fmac.append(d['macro'])
                    fwgh.append(d['weighted'])
                fmicro.append(statistics.mean(fmic))
                fmacro.append(statistics.mean(fmac))
                fweight.append(statistics.mean(fwgh))
            if fmicro:
                finalF1Score['micro']=statistics.mean(fmicro)
                print(finalF1Score['micro'])
            if fmacro:
                finalF1Score['macro']=statistics.mean(fmacro)
                print(finalF1Score['macro'])
            if fweight:
                finalF1Score['weighted']=statistics.mean(fweight)
                print(finalF1Score['weighted'])
            
            return (finalNMI,finalF1Score)
        
    @staticmethod
    def measureDBI(inputData,inputLabel):
        return davies_bouldin_score(inputData,inputLabel)
    
    @staticmethod
    def measureF1Score(yTrue=None,yPred=None):
        f1Macro=f1_score(yTrue, yPred, average='macro',labels=np.unique(yPred))
        f1Micro=f1_score(yTrue, yPred, average='micro',labels=np.unique(yPred))
        f1Weighted=f1_score(yTrue, yPred, average='weighted',labels=np.unique(yPred))
        return {'macro':f1Macro,'micro':f1Micro,'weighted':f1Weighted}    