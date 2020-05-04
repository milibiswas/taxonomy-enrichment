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

import getopt      
import sys
import os
import subprocess
from context import *

def main(param1,param2,param3):
    
    print('----------------------------------------------------------------------')
    print('             Starting the enhancing taxonomy program....              ')
    print('----------------------------------------------------------------------')
    
    inSavedModel='./src/data/output/'+param3+'.model'
    
    genvec=an.genvec(inSavedModel)
    
    # Here we need the corpus (list of list of words etc) from new data
    
    data=pp.Data(param1,param2)
    data.prepare()
    
    corpus=data.outputCorpusTokenized
    
    # Retraining the model using transfer learning
    genvec.retrain(corpus)
    
    # Get the vectors based on list of keywords
    
    keywordsList=data.outputKeys
    
    wordVecList=genvec.getWordVector(keywordsList)
    
    # Now enhance the taxonomy
    
    tree = an.taxonomy(genvec.model)
    
    # Enahncing the tree

    tree.process(wordVecList,'/Users/milibiswas/Desktop/json_data.txt')
    

    print('------------------------ End of the program --------------------------')
       
    
if __name__=='__main__':
    
    try:
        argv=sys.argv[1:]
        opts, args = getopt.getopt(argv, 'c:k:d:h', ['corpus_data=','keyword_data=','data_domain=','help'])
        
        for option,value in opts:
            if option in ['-c','--corpus_data']:
                param1=value
            elif option in ['-k','--keyword_data']:
                param2=value
            elif option in ['-d','--data_domain']:
                param3=value
            elif option in ['--help','-h']:
                print("<Usage> : python run.py [-c|--corpus_data=] <corpus data> [-k|=keyword_data] <keyword data> [-d|=data_domain] <data_domain>")
                print("<Example> : python run.py -c corpus.txt -k keyword.txt -d 'dblp' ")
                sys.exit(0)
            else:
                print("<Usage> : python run.py [-c|--corpus_data=] <corpus data> [-k|=keyword_data] <keyword data> [-d|=data_domain] <data_domain>")
    except getopt.GetoptError as err:
        print("<Usage> : python run.py [-c|--corpus_data=] <corpus data> [-k|=keyword_data] <keyword data> [-d|=data_domain] <data_domain>")
        print(str(err))
        sys.exit(1)
        

    from incremental import addnode as an
    from incremental import preprocess as pp 
     
    main(param1,param2,param3)