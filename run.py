#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =======================================================================================================
# 
#       Name : run.py
#       Description: To run the software package
#       Created by : Mili Biswas
#       Created on: 21.02.2020
#
#       Dependency : Software needs to be installed first
#
#       Parameters:
#               1 => corpus name (e.g. amazon_review, zalando, etc)
#               2 => taxonomy algorithm name (taxogen, hlda, hpam, ncrp, nole,noac,hclus)
#               3 => % of data to be processed (values between 0-1, 1 being 100%)
#               4 => data flush indicator (y being yes, n being no)
#
#
# ========================================================================================================

import getopt      
import sys
import os
import subprocess
from context import *
from preprocess import DataClean as dc
from hypertree import json_builder as jb
from preprocess import FastText as ft
#from preprocess import LangModel as lm
from evaluation import evaluate as ev  

        
def main(param1,param2,param3):
      
    tmpFilePath='./src/data/tmp/'
       
    # ==========================
    #  Instatiate the object
    # ==========================
            
    
    if param1 == 'amazon_fashion':
        groundTruth='./src/data/input/ground_truth/amazon_fashion_ground_truth.txt'
        maxLevel=3
        clusterInfo=[3,
                     5,0,0,0,0,0,
                     5,0,0,0,0,0,
                     5,0,0,0,0,0,
                     ]
        data=dc.Data(param1,param3,groundTruth)
        data.prepare()
    elif param1 == 'zalando_fashion':
        data=dc.Data(param1,param3)
        data.prepare()
    elif param1 == 'bbc':
        maxLevel=2
        clusterInfo=[2,
                       2,
                         1,
                       1,
                         1]
        data=dc.Data(param1,param3)
        data.prepare()
    elif param1 == 'dblp':
         maxLevel=1
         clusterInfo=[7,0,0,0,0,0,0,0
                     ]
         data=dc.Data(param1,param3)
         data.prepare()
    elif param1 == '20newsgroup':
        sys.exit('Under development')
        data=dc.Data(param1,param3)
        data.prepare()
    else:
        print('[Error]: The dataset name {} is not recognized'.format(param1))
        print('[Info]: The valid dataset names are {} , {} , {} , {} , {}'.format('amazon_fashion', 'zalando_fashion', 'bbc', 
              'dblp', '20newsgroup'))
        sys.exit(1)
        
        
    # Preprocess the raw data
    
    # ========================================
    #  Input word embeddings generation (w2v) 
    # ========================================
    
    print('[Info]: Running FastText for getting initial set of embeddings')
    model=ft.FastText(corpus=data.outputCorpusTokenized,window=5,min_count=5,iter=5,sample=1e-4)
    model.train()
    
    # Model is saved to be used during incremental process
    # Model saved file name is as param1.model where param1 = 'dblp' or 'amazon_review' etc.
    
    model.save('./src/data/output/'+param1+'.model')
    print('[Info]: Model is saved as ','./src/data/output/'+param1+'.model')
    
    # ========================================
    #  Keywords & Corpus generation
    # ========================================
    
    model.saveWordEmbeddings(path=os.path.join(tmpFilePath,'embeddings.txt'),keywordList=set(data.outputKeys))
    model.savekeywords(path=os.path.join(tmpFilePath,'keywords.txt'),keywordList=set(data.outputKeys))
    model.saveCorpus(path=os.path.join(tmpFilePath,'papers.txt'))
    print('[Info]: Files are saved at ',tmpFilePath)
    
    
    # ========================================
    #  Running the Taxonomy algorithm
    # ========================================
    
    
    def runTaxonomyAlgoritm(algoname, tmpAbsolutePath,percentData):
        corpusName=param1+'_'+str(percentData)
        if algoname in ['taxogen','taxogen_nole','taxogen_noac','hclus']:
            print('[Info]: Running {} algoritm'.format(algoname))
            os.chdir('./src/code/taxonomy_algorithm/'+algoname+'/code/')
            print('[Info]: Calling the subprocess run.sh')
            subprocess.call([  
                               './run.sh', 
                               tmpAbsolutePath,
                               '../data/',
                               corpusName,
                               str(clusterInfo),
                               str(maxLevel)
                             ])
    
            #=================================================
            #  Tree generation (JSON) for Hypertree Visualtion
            #=================================================
           
            print('[Info]: Calling json builder for generating JSON file')
            
            visObj=jb.JsonBuild()
            visObj.process('../data/'+corpusName+"/our-l3-0.15")
            os.chdir(dname)    
            visObj.createJavaScriptFile(path='./src/code/hypertree/Visualisation'+'/json_data.js')
            data.setJsonData(visObj.dataDict)
            
            print('[Info]: The algorithm {} is completed successfully'.format(algoname))
            
        
        elif algoname in ['hlda','hpam']:
            print('[Info]: Running hlda algoritm')
            os.chdir('./src/code/taxonomy_algorithm/'+algoname)
            subprocess.call([
                               "python3",
                               "./hlda.py",
                               tmpAbsolutePath+'/papers.txt',
                               tmpAbsolutePath+'/keywords.txt',
                               "2",
                               './../../hypertree/Visualisation'+'/json_data.js'
                             ])
            os.chdir(dname)
            
        elif algoname == 'ncrp':
            print('==========================================================================')
            print('[Info]: Running Chinese Restaurant Process (nCRP) algoritm')
            os.chdir('./src/code/taxonomy_algorithm/'+algoname)
            subprocess.call([
                              "python3",
                              "./generate_taxonomy.py",
                              tmpAbsolutePath,
                              param1,
                              str(clusterInfo),
                              str(maxLevel),
                              './../../hypertree/Visualisation/json_data.js'
                            ])
            os.chdir(dname)
            print('[Info]: The algorithm {} is completed successfully'.format(algoname))
            
        else:
            print('[Warning]: The algorithm {} is not included or recognized'.format(algoname))
            sys.exit(1)
    
    # ===============================================
    #     Prepartion for running taxonomy algorithm
    # ===============================================
            
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    tmpAbsolutePath=os.path.abspath(tmpFilePath)
    runTaxonomyAlgoritm(param2,tmpAbsolutePath,param3)
    
    # ===============================================
    #     Evaluation
    # ===============================================
    if data.gtDataList:
        try:
            eval_obj= ev.Evaluation(data.gtDataList,data.jsonDict)
            eval_obj.process()
            nmi_score=ev.Evaluation.measureNMI(eval_obj.gtList,eval_obj.pathList, data.keywords2label)
            #print(ev.Evaluation.measureF1Score())
            print(nmi_score)
        except Exception as err:
            print('[Error]: '+str(err))
    else:
        print('[Warning]: Ground truth was not evailable, so evaluation did not run')
    
    
if __name__=='__main__':
    
    try:
        print('------------------------------------------------------')
        print('                Starting the program....              ')
        print('------------------------------------------------------')
        argv=sys.argv[1:]
        opts, args = getopt.getopt(argv, 'd:a:n:h', ['dataset_name=','algorithm_name=','percent_of_data=','help'])
        
        for option,value in opts:
            if option in ['-d','--dataset_name']:
                param1=value
            elif option in ['-a','--algorithm_name']:
                param2=value
            elif option in ['-n','--percent_of_data']:
                param3=float(value)
            elif option in ['--help','-h']:
                print("<Usage> : python run.py [-d|--dataset_name=]<Dataset Name> [-a|algorithm_name=]<algorithm name> [-n|--percent_of_data]<percent of data to be processed>")
                print("<Example> : python run.py -d 'amazon_fashion' -n 0.01 -a 'taxogen")
                sys.exit(0)
            else:
                print("<Usage> : python run.py [-d|--dataset_name=]<Dataset Name> [-a|algorithm_name=]<algorithm name> [-n|--percent_of_data]<percent of data to be processed>")
    except getopt.GetoptError as err:
        print("<Usage> : python run.py [-d|--dataset_name=]<Dataset Name> [-a|algorithm_name=]<algorithm name> [-n|--percent_of_data]<percent of data to be processed>")
        print(str(err))
        sys.exit(1)


    main(param1,param2,param3)
    

    # =============================
    #     END of the program
    # =============================

