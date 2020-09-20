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
#               3 => % of data to be processed (values between 0-100)
# ========================================================================================================

import pickle
import getopt
import sys
import os
import subprocess
import yaml
from context import *
from preprocess import DataClean as dc
from hypertree import json_builder as jb
from preprocess import FastText as ft
#from preprocess import LangModel as lm
from evaluation import evaluate as ev

def makeDirForSavingModel(dirName):
    try:
        if os.path.exists(dirName) and os.path.isdir(dirName):
            if not os.listdir(dirName):
                pass
            else:
                for file in os.listdir(dirName):
                    os.remove(os.path.join(dirName,file))
        else:
            os.mkdir(dirName)
    except Exception as e:
        print(e)
        sys.exit(-1)
        
def configFileRead(filePath):
    try:
        with open(filePath,'r') as f:
            retVal=yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print('[Error]: Problem processing the config.yml file')
        print('[Error]:',e)
        sys.exit(-1)
    return retVal

def main(param1,param2,param3):
    
    # Loading the configurations and parameters details
    config=configFileRead('./config.yml')
    
    # cleans up the directory where trained model will be saved
    modelSavePath=config['pathvariable']['modelSavePath']
    makeDirForSavingModel(modelSavePath)
    
    # Different File Path loading
    tmpFilePath=config['pathvariable']['tmpFilePath']
    hypertreeJsonFile=config['pathvariable']['hypertreeJsonFile']
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    tmpAbsolutePath=os.path.abspath(tmpFilePath)
    hypertreeAbsolutePathJsonFile=os.path.abspath(hypertreeJsonFile)
    
    #  Instatiate the object to call different algorithms
    if param1 == 'amazon_fashion':
        
        groundTruth=os.path.join(config['pathvariable']['groundTruthPath'],config[param1]['otherparam']['groundTruthFileName'])
        maxLevel=config[param1]['otherparam']['maxlevel']
        clusterInfo=list(config[param1]['clusterinfo'].values())
        sample=float(config[param1]['otherparam']['sample'])
        window=config[param1]['otherparam']['window']
        min_count=config[param1]['otherparam']['min_count']
        iter=config[param1]['otherparam']['iter']
        size=config[param1]['otherparam']['size']
        data=dc.Data(param1,param3,groundTruth)
        data.prepare()
        
    elif param1 == 'bbc':
        
        groundTruth=os.path.join(config['pathvariable']['groundTruthPath'],config[param1]['otherparam']['groundTruthFileName'])
        maxLevel=config[param1]['otherparam']['maxlevel']
        clusterInfo=list(config[param1]['clusterinfo'].values())
        sample=float(config[param1]['otherparam']['sample'])
        window=config[param1]['otherparam']['window']
        min_count=config[param1]['otherparam']['min_count']
        iter=config[param1]['otherparam']['iter']
        size=config[param1]['otherparam']['size']
        data=dc.Data(param1,param3,groundTruth)
        data.prepare()
        
    elif param1 == 'dblp':
        
        groundTruth=os.path.join(config['pathvariable']['groundTruthPath'],config[param1]['otherparam']['groundTruthFileName'])
        maxLevel=config[param1]['otherparam']['maxlevel']
        clusterInfo=list(config[param1]['clusterinfo'].values())
        sample=float(config[param1]['otherparam']['sample'])
        window=config[param1]['otherparam']['window']
        min_count=config[param1]['otherparam']['min_count']
        iter=config[param1]['otherparam']['iter']
        size=config[param1]['otherparam']['size']
        data=dc.Data(param1,param3,groundTruth)
        data.prepare()
        
    else:
        
        print('[Error]: The dataset name {} is not recognized'.format(param1))
        print('[Info]: The valid dataset names are {} , {} , {} , {} , {}'.format('amazon_fashion', 'bbc',
              'dblp'))
        sys.exit(1)
    

    #  Initial word embeddings generation (w2v)
    print('[Info]: Running word2Vec for getting initial set of embeddings')
    model=ft.FastText(corpus=data.outputCorpusTokenized,window=window,min_count=min_count,iter=iter,sample=sample)
    model.train()
    
    # Model is saved to be used during incremental process
    # Model saved file name is as param1.model where param1 = 'dblp' or 'amazon_review' etc.
    if param2!='taxogen' or param2!='taxogen_noac':
        model.save(os.path.join(modelSavePath,param1+'.model'))
        print('[Info]: Model is saved as ',os.path.join(modelSavePath,param1+'.model'))
    

    #  Keywords & Corpus generation
    model.saveWordEmbeddings(path=os.path.join(tmpFilePath,'embeddings.txt'),keywordList=set(data.outputKeys))
    model.savekeywords(path=os.path.join(tmpFilePath,'keywords.txt'),keywordList=set(data.outputKeys))
    model.saveCorpus(path=os.path.join(tmpFilePath,'papers.txt'))
    print('[Info]: Initial corpus, embeddings and keywords files are saved at ',tmpFilePath)
    

    #  Running the Taxonomy algorithm
    def runTaxonomyAlgoritm(algoname, tmpAbsolutePath,percentData):
        corpusName=param1+'_'+str(percentData)
        
        if algoname in ['taxogen_nole','hclus']:
            
            print('==========================================================================')
            print('[Info]: Running {} algoritm'.format(algoname))
            print('==========================================================================')
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
            #  Tree generation (JSON) for Hypertree Visualtion
            print('[Info]: Calling json builder for generating JSON file')
            visObj=jb.JsonBuild()
            visObj.process('../data/'+corpusName+"/our-l3-0.15")
            os.chdir(dname)
            visObj.createJavaScriptFile(path=str(hypertreeAbsolutePathJsonFile))
            data.setJsonData(visObj.dataDict)
            print('[Info]: The algorithm {} is completed successfully'.format(algoname))
        
        elif algoname in ['taxogen','taxogen_noac']:
            
            print('==========================================================================')
            print('[Info]: Running {} algoritm'.format(algoname))
            print('==========================================================================')
            os.chdir('./src/code/taxonomy_algorithm/'+algoname+'/code/')
            print('[Info]: Calling the subprocess run.sh')
            subprocess.call([
                               './run.sh',
                               tmpAbsolutePath,
                               '../data/',
                               corpusName,
                               str(clusterInfo),
                               str(maxLevel),
                               str(size),
                               str(sample),
                               str(window),
                               str(min_count),
                               str(iter)
                             ])
            #  Tree generation (JSON) for Hypertree Visualtion
            print('[Info]: Calling json builder for generating JSON file')
            visObj=jb.JsonBuild()
            visObj.process('../data/'+corpusName+"/our-l3-0.15")
            os.chdir(dname)
            visObj.createJavaScriptFile(path=str(hypertreeAbsolutePathJsonFile))
            data.setJsonData(visObj.dataDict)
            print('[Info]: The algorithm {} is completed successfully'.format(algoname))
            
        elif algoname == 'ncrp':
            
            print('==========================================================================')
            print('[Info]: Running Chinese Restaurant Process (nCRP) algoritm')
            print('==========================================================================')
            os.chdir('./src/code/taxonomy_algorithm/'+algoname)
            subprocess.call([
                              "python3",
                              "./generate_taxonomy.py",
                              tmpAbsolutePath,
                              param1,
                              str(clusterInfo),
                              str(maxLevel),
                              str(hypertreeAbsolutePathJsonFile)
                            ])
            os.chdir(dname)
            print('[Info]: The algorithm {} is completed successfully'.format(algoname))
            
        else:
            print('[Warning]: The algorithm {} is not included or recognized'.format(algoname))
            sys.exit(1)
    # ===============================================
    #     Running taxonomy algorithm
    # ===============================================
    runTaxonomyAlgoritm(param2,tmpAbsolutePath,param3)
    # ===============================================
    #     Evaluation of algorithms
    # ===============================================
    try:
        if param2 != 'ncrp':
            evalObj= ev.Evaluation('./src/data/output/taxonomy.json',groundTruth)
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
        else:
            #file = open(os.path.join(tmpFilePath,'ncrp_datadict.pkl'), 'rb')
            #dump information to that file
            #dataDict = pickle.load(file)
            #close the file
            #file.close()
            evalObj= ev.Evaluation('./src/data/output/taxonomy.json',groundTruth)
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
                
    except Exception as err:
        print('[Error]: '+str(err))
    
    print('                 ')
    print('------------------------ End of the program --------------------------')

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