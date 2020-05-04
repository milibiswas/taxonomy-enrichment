#####################################################
#
#    file name : generate_taxonomy.py
#    created by: Mili Biswas (MSc - Comp. Sc, UNIFR)
#    creation Dt: 28th Nov 2019
#
#####################################################

import create_network as cn
import eval_modified as ev
import build_cluster as bc
import sys
import os
import pickle
import ast

dirPath='./data/'

if __name__=='__main__':
    try:
        argv=sys.argv[1:]
        if len(argv)==5:
            
            tempDirPath=argv[0]
            datasetName=argv[1]
            clusterInfo=ast.literal_eval(argv[2])
            maxLevel=int(argv[3])
            jsonFilePath=argv[4]
            
            if datasetName=='bbc':
                '''
                    This is for BBC dataset prep for ncrp algorithm
                '''
                
                with open(os.path.join(tempDirPath,datasetName+'_target_label_modified.pkl'),'rb') as fin:
                    target_label=pickle.load(fin)
                
                corpus={}
                cnt=0
                with open(os.path.join(tempDirPath,'papers.txt'),'r') as fin:
                    try:
                        for key,line in enumerate(fin):
                            if line.strip('\n'):
                                if target_label[key] not in corpus:
                                    corpus[target_label[key]]=[]
                                corpus[target_label[key]].append(line.strip('\n'))
                                cnt = cnt +1
                    except Exception as err:
                        print('[Error]: ',str(err))
                        
                cn.main_title(corpus,dirPath)
                ev.embedding(dirPath, ev.NetData(dirPath, 'word_cooc'), ev.AlgoNCRP, 'db')
                clustObj=bc.cluster(dirPath,clusterInfo,maxLevel,os.path.join(tempDirPath,'keywords.txt'),jsonFilePath)
                clustObj.prepare()
                
            elif datasetName=='dblp':
                '''
                    This is for dblp dataset prep for ncrp algorithm
                '''
                
                with open(os.path.join(tempDirPath,datasetName+'_target_label_modified.pkl'),'rb') as fin:
                    target_label=pickle.load(fin)
                
                corpus={}
                cnt=0
                with open(os.path.join(tempDirPath,'papers.txt'),'r') as fin:
                    try:
                        for key,line in enumerate(fin):
                            if line.strip('\n'):
                                if target_label[key] not in corpus:
                                    corpus[target_label[key]]=[]
                                corpus[target_label[key]].append(line.strip('\n'))
                                cnt = cnt +1
                    except Exception as err:
                        print('[Error]: ' + str(err))
                        
                cn.main_title(corpus,dirPath)
                ev.embedding(dirPath, ev.NetData(dirPath, 'word_cooc'), ev.AlgoNCRP, 'db')
                clustObj=bc.cluster(dirPath,clusterInfo,maxLevel,os.path.join(tempDirPath,'keywords.txt'),jsonFilePath)
                clustObj.prepare()
                
            else:
                
                '''
                    This part of the code is for those dataset where document group is not known unlike BBC and DBLP.
                    Therefore, for these datasets, ncrp process will run for each blogs as distinct document.
                '''
                
                
                corpus={}
                with open(os.path.join(tempDirPath,'papers.txt'),'r') as fin:
                    try:
                        for key,line in enumerate(fin):
                            if line.strip('\n'):
                                if key not in corpus:
                                    corpus[key]=[]
                                corpus[key].append(line.strip('\n'))
                    except Exception as err:
                        print('[Error]: '+ str(err))                
                cn.main_title(corpus,dirPath)
                ev.embedding(dirPath, ev.NetData(dirPath, 'word_cooc'), ev.AlgoNCRP, 'db')
                clustObj=bc.cluster(dirPath,clusterInfo,maxLevel,os.path.join(tempDirPath,'keywords.txt'),jsonFilePath)
                clustObj.prepare()
            
        else:
            sys.exit('[Error]: Wrong parameter is given in generate_taxonomy.py module in nethiex algorithm')
            
    except Exception as err:
        sys.exit('[Error]: '+str(err))