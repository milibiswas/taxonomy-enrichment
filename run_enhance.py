#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =======================================================================================================
# 
#       Name : run_enhance.py
#       Description: To run the software package
#       Created by : Mili Biswas
#       Created on: 21.02.2020
#
#       Dependency : Software needs to be installed first
#       Parameters:
#                  param1 -> Corpus
#                  param2 -> Keywords
# ========================================================================================================
import yaml
import pandas as pd
import numpy as np
import getopt      
import sys
import os
import subprocess
from context import *
from incremental import preprocess as pp
from incremental import addnode as an
from evaluation import evaluate as ev 

def randomDataSlice(data,volume):
    '''
         Parameters:
                    data -> input data (must be a list)
                    volume -> % of data volume that needs to be sliced and return (must be between 0 and 100)
         Description:
                    This function randomly selects data from the input list and returned the sliced information
                    that were given via volume parameter.
    '''
    try:
        if volume<=0 or volume>100:
            raise Exception
    except Exception:
        print('[Error]: percent of data is given wrong. Value must belong to (0,100]')
        sys.exit(1)
    
    try:
        v=volume/100
        dataSeries=pd.Series(data)
        upperLimit=dataSeries.count()
        lowerLimit=int(upperLimit*v)
        index=np.random.choice(upperLimit, lowerLimit,replace=False)
        modDataSeries=dataSeries.loc[ index ]
    except Exception as e:
        print('[Error]:',e)
        
    return modDataSeries
    
def configFileRead(filePath):
    try:
        with open(filePath,'r') as f:
            retVal=yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print('[Error]: Problem processing the config.yml file')
        print('[Error]:',e)
        sys.exit(-1)
    return retVal
    
    

def main(param1,param2=100):
    
    print('----------------------------------------------------------------------')
    print('          Starting the enhancing taxonomy program algorithm           ')
    print('----------------------------------------------------------------------')
    
    # Loading the configurations and parameters details
    config=configFileRead('./config_incremental.yml')
    
    # Loading different necessary paths
    if param1 =='amazon_fashion':
        corpusFile=config[param1]['pathvariable']['corpusFile']
        keywordsFile=config[param1]['pathvariable']['keywordsFile']
        modelSavePath=config[param1]['pathvariable']['modelSavePath']
        groundTruthFile=config[param1]['pathvariable']['groundTruthFile']
        enhanceTxonomyFile=config[param1]['pathvariable']['enhanceTxonomyFile']
    elif param1 =='bbc':
        corpusFile=config[param1]['pathvariable']['corpusFile']
        keywordsFile=config[param1]['pathvariable']['keywordsFile']
        modelSavePath=config[param1]['pathvariable']['modelSavePath']
        groundTruthFile=config[param1]['pathvariable']['groundTruthFile']
        enhanceTxonomyFile=config[param1]['pathvariable']['enhanceTxonomyFile']
    elif param1 =='dblp':
        corpusFile=config[param1]['pathvariable']['corpusFile']
        keywordsFile=config[param1]['pathvariable']['keywordsFile']
        modelSavePath=config[param1]['pathvariable']['modelSavePath']
        groundTruthFile=config[param1]['pathvariable']['groundTruthFile']
        enhanceTxonomyFile=config[param1]['pathvariable']['enhanceTxonomyFile']
    else:
        print('data_domain (-d) value is not found <given> :', param1)
        sys.exit(-1)
    
    # Preprocessing object is initialized
    data=pp.Data(corpusFile,keywordsFile)
    
    # prepare() method of preprocess object instance triggers the data preprocess before it can be
    # sent to re-train.
    data.prepare()
    
    # tokenized preprocessed tokenized data is returned to variable
    corpus=randomDataSlice(data.outputCorpusTokenized,param2)
    print('[Info]: The corpus data count',corpus.count())
    
    
    # This is genvec object initialization using saved model
    genvec=an.genvec(modelSavePath)
    
    # Retraining the model using transfer learning
    genvec.retrain(corpus)
    
    # list of keywords from preprocessed objcet instance
    keywordsList=data.outputKeys
    
    # Get the vectors based on list of keywords
    wordVecList=genvec.getWordVector(keywordsList)
    
    '''
        =====================================================
        From here, we'll enhance the existing taxonomy First 
        =====================================================
    '''
    
    # Instantiate the taxonomy class object using the retrained model
    
    tree = an.taxonomy(genvec.model)
    
    # Enhance the existing taxonomy (tree) based on the similarity score technique
    
    #print(wordVecList)

    tree.process(wordVecList,'./src/data/output/taxonomy.json')
    
    # ===============================================
    #     Evaluation of algorithms
    # ===============================================
    
    try:
        evalObj= ev.Evaluation(enhanceTxonomyFile,groundTruthFile)
        evalObj.process()
        NMIScore=ev.Evaluation.measureNMI(evalObj.groundTruth,evalObj.pathList, evalObj.keywords2label)
        nmi,f1=NMIScore
        print('=====================================================')
        print('                       Results                       ')
        print('=====================================================')
        print('Level            NMIScore')
        print('-----            ------------------------')
        for key,val in nmi.items():
            print(key,'              ',val)
            
        print('                 ')
        
        print('                 Average F1 Score')
        print('                 ----------------')
        for key,val in f1.items():
            print(key,'          ',val)
            
    except Exception as e:
        print('[Error]:',e)
    
    print('                 ')
    print('------------------------ End of the program --------------------------')
       
    
if __name__=='__main__':
    
    try:
        argv=sys.argv[1:]
        opts, args = getopt.getopt(argv, 'd:n:h', ['corpus_data=','keyword_data=','data_domain=','percent_of_data=','help'])
        
        for option,value in opts:
            if option in ['-d','--data_domain']:
                param1=value
            elif option in ['-n','--percent_of_data']:
                param2=float(value)
            elif option in ['--help','-h']:
                print("<Usage> : python run_enhance.py [-d|--data_domain=] <data_domain> [-n|--percent_of_data=] <% of input data> ")
                print("<Example with 10% data> : python run_enhance.py -d 'dblp' -n 10")
                sys.exit(0)
            else:
                print("<Usage> : python run_enhance.py [-d|--data_domain=] <data_domain> [-n|--percent_of_data=] <% of input data> ")
                print("<Example with 10% data> : python run_enhance.py -d 'dblp' -n 10")
                sys.exit(0)
                
        #============================================= 
        # Calling the main function
        #=============================================  
        
        main(param1,param2)
    
    except getopt.GetoptError as e:
        print("<Usage> : python run_enhance.py [-d|--data_domain=] <data_domain> [-n|--percent_of_data=] <% of input data> ")
        print("<Example with 10% data> : python run_enhance.py -d 'dblp' -n 10")
        print(e)
        sys.exit(1)
    except NameError as e:
        print('Internal error message:',e)
        print('Please check the parameters supplied as mentioned in <Usage>!')
        print("<Usage> : python run_enhance.py [-d|--data_domain=] <data_domain> [-n|--percent_of_data=] <% of input data> ")
        print("<Example with 10% data> : python run_enhance.py -d 'dblp' -n 10")
        sys.exit(1)
     
    