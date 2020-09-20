#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    File Name: preprocess.py 
    Purpose: This script is made for preprocessing the dataset.
    Created By: Mili Biswas (MSc. Computer Sc., UNIFR, CH)
    Date: 4th Feb 2020

'''

#---------------------------------
# Import required python modules
#---------------------------------
from lxml import etree as ElementTree
from contextlib import redirect_stdout
import sys
from zipfile import ZipFile
import os
import gzip
from collections import deque
import re
import collections
import pickle
import urllib.request
#import xml.etree.ElementTree

import pandas as pd
import numpy as np
from sklearn.datasets import load_files

import nltk
from nltk.stem import WordNetLemmatizer
from nltk import  FreqDist

from preprocess import keyword_extract as ke

with redirect_stdout(open(os.devnull, "w")):
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

en_stop = set(nltk.corpus.stopwords.words('english'))


#---------------------------------
# Custom Error Classes
#---------------------------------


class Error(Exception):
    pass

class CustomErrorForCorpus(Error):
    pass

class CustomErrorForKeyWords(Error):
    pass

class CustomErrorDataVolumeCheck(Error):
    pass

#---------------------------------
# Class that preprocess the corpus
#---------------------------------

class Data(object):
    
    def __init__(self,dataSetName,dataVolume=1.0,groundTruth=None):
        
        # Input Parameters
        self.dataSetName=dataSetName
        self.inputFile='./src/data/input/corpus/'+dataSetName+'_corpus.txt'
        self.inputKeyFile='./src/data/input/keywords/'+dataSetName+'_keywords.txt'
        self.tempPath='./src/data/tmp/'
        self.volume=float(dataVolume)
        self.url={
                'amazon_review':'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/AMAZON_FASHION_5.json.gz',
                'bbc':'http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip',
                'dblp':'http://dblp.org/xml/dblp.xml.gz',
                '20newsgroup':'NA'}
        
        # Output Parameters
        
        self.outputData=None
        self.outputKeys=None
        self.outputCorpusTokenized=None
        self.outputProcessedCorpus=None
        self.groundTruth=groundTruth
        self.gtDataList=None
        self.keywords2label={}
        self.jsonDict=None
        
        # Other Parameters
        
        self.tfidfScore=None
        self.lem=WordNetLemmatizer()
        self.stopWord = self.buildStopWords(self.readStopWords('./src/data/input/other/en_stopwords.txt'))
        #self.stopWord = self.buildStopWords(en_stop)
        self.most_commonn_word = None
        
        self.targetLabel=None
        
        
        # Methods in init()
        try:
            if self.volume<=0 or self.volume>100:
                raise CustomErrorDataVolumeCheck
        except CustomErrorDataVolumeCheck:
            print('[Error]: percent of data is given wrong. Value must belong to (0,100]')
            sys.exit(1)
        
        
    def downloadData(self,filePath,url):
        '''
           This function will download the dataset 
        '''
        
        # =============================================================================
        fileName=url.split('/')[-1]
        downloadFile=filePath+fileName
        # =============================================================================
        
        print('[Info]: Beginning file download')
            
        # =============================================================================
        #url = 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/AMAZON_FASHION_5.json.gz'
        urllib.request.urlretrieve(url,downloadFile)
        # =============================================================================
        print('[Info]: File download completed successfully')
        return fileName
    
    @staticmethod
    def readXML(xmlPath):
        
        categoryList=[  'BIBM',
                        'RECOMB',
                        'INFOCOM',
                        'SIGCOMM',
                        'SC',
                        'ISCA',
                        'EUROCRYPT',
                        'CRYPTO',
                        'DCC',
                        'CVPR',
                        'ICCV',
                        'ACL',
                        'COLING'
                        ]
        parser=ElementTree.XMLParser(dtd_validation=True)
        #root = xml.etree.ElementTree.parse(xmlPath).getroot()
        root = ElementTree.parse(xmlPath, parser).getroot()
        articles = []
        for category in categoryList:
            cnt = 0
            for paper in root.iter('inproceedings'):
                booktitle = next(paper.iter('booktitle')).text
                
                #year = int(next(paper.iter('year')).text)     # This has been commented by Mili (not used in the filter)

                # TODO -- the year & book category can be made parameterized
                
                if booktitle == category:       # year has been removed from filter
                    articles.append(paper)
                    cnt += 1
        return articles
    
    @staticmethod
    def parse(path):
        g = open(path, 'r')
        for l in g:
            yield eval(l)
            
    @staticmethod
    def getDF(path):
        i = 0
        df = {}
        for d in Data.parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')
    
    def fileExist(self,file):
        if os.path.exists(file):
            return True
        return False
    
    def loadFile(self,file):
        
        inpData=[]
        
        with open(file,'r',encoding='utf8') as fin:
            for line in fin:
                if line:
                    inpData.append(line.strip('\n'))
        return inpData
    
    def setJsonData(self,data):
        self.jsonDict=data
        
    
    def __setOutputData(self,inputFile):
        '''
            This function will set the self.outputData
        '''
        text=[]
        rawCorpus=self.loadFile(self.inputFile)
        for line in rawCorpus:
            text.append(self.preprocessText(line))
        self.outputData=text
        
        return None
    
    def __setKeywordData(self,inputKeyFile):
        keys=[]
        rawKeywords=self.loadFile(inputKeyFile)
        for line in rawKeywords:
            tmp=self.preprocessWord(line)
            keys.append(tmp.replace(' ','_'))
            self.outputKeys=keys
            
        return None
    
    def __processGroundTruth(self,):
        if not self.groundTruth:
            print('[Warning]: Ground Truth not provided')
        else:
            if self.fileExist(self.groundTruth):
                gtDataList=[]
                with open(self.groundTruth,'r') as fin:
                    for line in fin:
                        recList=line.strip('\n').split(',')
                        gtDataList.append(recList[:-1])
                        self.keywords2label[recList[-2]] = recList[-1]
                self.gtDataList=gtDataList
            else:
                print("[Error] Ground Truth file {} not found or can't be accessed".format(self.groundTruth))
                
                    
                
    def getFileNameFromPath(self,path):
        return os.path.basename(path)
    
    def readStopWords(self,inpFile):
        stopWord=[]
        with open(inpFile,'r') as fin:
            for line in fin:
                stopWord.append(line.strip('\n'))
        return stopWord
        
    def buildStopWords(self,stopWord):    
        wordList=[self.preprocessWord(w) for w in stopWord]
        return wordList
    
    def subs(self,keyword):
        q = deque()
        l_w = keyword.split('_')
        l_e= ''
        r_e = ''
        cnt = 0
        for w in l_w:
            q.append(w)
        while q:
            front=q.popleft()
            if cnt == 0:
                l_e = front
                r_e = front
            else:
                l_e+="\s+"+front
                r_e+="_"+front
            cnt+=1
        return {"left":l_e,"right":r_e}
    
    def lemmatize(self,w):
        w1=self.lem.lemmatize(w,pos='v')
        w2=self.lem.lemmatize(w1,pos='n')
        w3=self.lem.lemmatize(w2,pos='a')
        return w3
    
    
    def removeStopWords(self,document):
        tokens = document.split()
        tokens = [word for word in tokens if word not in self.stopWord and len(word)>2]
        processed_doc = ' '.join(tokens)
        return processed_doc
    
    
    def compute_tfidf(self,tokenizedCorpus):    
        l=[]
        topics={}
        vocab=[]
        for key,val in enumerate(tokenizedCorpus):
            if key not in topics:
                topics[key]=None
            topics[key]=val
            vocab = vocab+val
        vocab=set(vocab)
        # Compute tf-idf.
        tf = dict((v, {}) for v in topics)
        for v in topics:
            for w in topics[v]:
                if w in tf[v]:
                    tf[v][w] += 1
                else:
                    tf[v][w] = 1
                    l.append(w)
                    
        c = collections.Counter(l)
        idf = dict((w, 0) for w in vocab)

        for w in vocab:
            '''cnt = 0
            for v in topics:
                if w in tf[v]: 
                    cnt += 1'''
            cnt=c[w]
            
            idf[w] = np.log(len(topics) / cnt)
            
        tf_idf = dict((v, {}) for v in topics)
        
        for v in topics:
            for w in topics[v]:
                tf_idf[v][w] = tf[v][w] * idf[w]
        self.tfidfScore=tf_idf
    
    
    def preprocessWord(self,word):
        # Remove all the special characters
        word = re.sub("[^a-zA-Z-']+", ' ', str(word))
        # remove all single characters
        word = re.sub(r'\s+[a-zA-Z]\s+', ' ', word)
        # Remove single characters from the start
        word = re.sub(r'\^[a-zA-Z]\s+', ' ', word)
        # Removing prefixed "'"
        word = re.sub(r"\s+'", '', word)
        # replacing multiple "-" into one "-"
        word = re.sub(r"\-+", '-',word)
        # removing "-" from beginning & end 
        word = re.sub(r"(^-+|-+$)", '', word)
        word = re.sub(r"-+\s+", ' ', word)
        word = re.sub(r"\s+-+", ' ', word)
        # Substituting multiple spaces with single space
        word = re.sub(r'\s+', ' ', word, flags=re.I)    
        # Converting to Lowercase
        word = word.lower()
        # Converting to Lowercase
        word =' '.join([self.lemmatize(w) for w in word.split()])
        return word
    
    def preprocessText(self,document):
        # Remove all the special characters
        document = re.sub("[^a-zA-Z-']+", ' ', str(document))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r"\s+'", '', document)    
        # replacing multiple "-" into one "-"
        document = re.sub(r"\-+", '-',document)
        # removing "-" from beginning & end 
        document = re.sub(r"(^-+|-+$)", '', document)
        document = re.sub(r"-+\s+", ' ', document)
        document = re.sub(r"\s+-+", ' ', document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        tokens = document.split()
        #tokens = [word for word in tokens]
        '''tokens = [word for word in tokens if word not in en_stop]'''
        #tokens = [word for word in tokens if len(word)>1]
        tokens = [self.lemmatize(word) for word in tokens]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    
    @staticmethod
    def keywordExtraction(
                 corpusName,
                 keywordFilePath,
                 topn=10
            ):
        try:
            ke.main(corpusName,keywordFilePath,topn)
            return None
        except Exception as err:
            print(str(err))
            sys.exit('Error in keywordExtraction() ')
        
        
    def processKeyWords(self,):
        '''
           processKeyWords() method preprocess keywords data.
           Each keyword (where mutliple words exist) is being
           joined by underscore ('_').
        '''
        inpKeyFileName=self.getFileNameFromPath(self.inputKeyFile)
        pickleKeyFileName='preprocess_'+inpKeyFileName+'.pkl'
            
        if self.fileExist(self.inputKeyFile):
            self.__setKeywordData(self.inputKeyFile)
            keysPicklePathDir=self.tempPath
                    
            if self.fileExist(keysPicklePathDir):
                pass
            else:
                os.mkdir(keysPicklePathDir)
                        
            with open(os.path.join(keysPicklePathDir,pickleKeyFileName),'wb') as fout:
                pickle.dump(self.outputKeys,fout)
            return None
        else:
            # The keywords will be extracted and saved
            try:
                Data.keywordExtraction(self.outputData,self.inputKeyFile,10)
                self.processKeyWords()
            except  CustomErrorForKeyWords:
                print('[Error]: Keywords process error')
                sys.exit(1)
    
    def processData(self,):
        '''
           processData() method preprocess raw corpus data.
           Every record i.e. each review within corpus is being 
           preprocessed where keywords are being harmonized (e.g. joined by underscore,
        l  ammetized etc.) before being passed to next stage.
        '''
        try:
            if self.fileExist(self.inputFile):
                self.__setOutputData(self.inputFile)
                
            else:
                # Download the file and create the text file
                self.processDownloadedCorpus(
                            self.downloadData(self.tempPath,self.url[self.dataSetName])
                            )
                self.processData() 
                
            return None
                
        except  CustomErrorForCorpus:
            print('[Error]: Preprocessing data')
            
    def processDownloadedCorpus(self,downloadFile):
        
        if self.dataSetName=='amazon_fashion':
            
            df = Data.getDF(os.path.join(self.tempPath,downloadFile))
            
            with open(self.inputFile,'w') as fin:
                   for review in list(df['reviewText']):
                       fin.write(review.strip('\n'))
                       fin.write('\n')
                       
            return None
                       
        elif self.dataSetName=='bbc':
            
            EXTRACT_DATA_DIR=self.tempPath
            
            with ZipFile(os.path.join(self.tempPath,downloadFile), 'r') as zipObj:
               zipObj.extractall(EXTRACT_DATA_DIR)
               bbcData = load_files(EXTRACT_DATA_DIR+'/bbc/', encoding="utf-8", decode_error="replace")
               
               with open(self.inputFile,'w') as fin:
                   for review in bbcData['data']:
                       tmp=''
                       lines = review.split("\n")
                       non_empty_lines = [line for line in lines if line.strip() != ""]
                       for s in non_empty_lines:
                           if s.strip('\n'):
                               tmp=tmp+s.strip('\n')+' '
                       fin.write(tmp)
                       fin.write('\n')
               
                
               # Keeping the target label. This is numpy array
               with open(os.path.join(self.tempPath,'bbc_target_label.pkl'),'wb') as fout:
                    pickle.dump(list(bbcData['target']),fout)
                               
            return None
        
        elif self.dataSetName=='dblp':
            #TODO
            print('[Info]: Unzipping the file started')
            xmlGzip = gzip.GzipFile(os.path.join(self.tempPath,downloadFile), 'rb')
            xmlReadObj = xmlGzip.read()
            xmlGzip.close()
            
            output = open(os.path.join(self.tempPath,"dblp.xml"), 'wb')
            output.write(xmlReadObj)
            output.close()
            print('[Info]: Unzipping the file completed')
            
            print('[Info]: Creating papers.txt file started')
            xmlFile=os.path.join(self.tempPath,"dblp.xml")
            topics = {}
            for paper in Data.readXML(xmlFile):
                tl = next(paper.iter('title')).text
                key = next(paper.iter('booktitle')).text
                if key not in topics:
                    topics[key] = []
                topics[key].append(tl)
                    
            with open(self.inputFile,'w',encoding='utf8') as fin:
                   for key,val in topics.items():
                       for title in val:
                           if title:
                               fin.write(title.strip('\n'))
                               fin.write(' ')
                       fin.write('\n')
            
            print('[Info]: Creating papers.txt file completed')
            
            # Keeping the target label. This is numpy array 
            with open(os.path.join(self.tempPath,'dblp_target_label.pkl'),'wb') as fout:
                    pickle.dump(list(topics.keys()),fout)
            
            return None
          
        
        elif self.dataSetName=='20newsgroup':
            #TODO
            return None
        else:
            sys.exit('Error in processDownloadedCorpus() - Dataset name does not exists : {}'.format(self.dataSetName))
    
    def wordTokenization(self,corpus):
        final_corpus = [self.removeStopWords(review) for review in corpus if review.strip() !='']
        word_tokenized_corpus = [review.split() for review in final_corpus]
        self.outputCorpusTokenized=word_tokenized_corpus
        
    def freq_dist(self,):
        l = sum(self.outputCorpusTokenized,[])
        fd = FreqDist(l)
        self.most_commonn_word=[t[0] for t in fd.most_common(1) if t[0] 
                                 not in self.outputKeys]
        
        new_list=[]
        
        for l in self.outputCorpusTokenized:
            tmp=[]
            
            for w in l:
                if w not in self.most_commonn_word:
                    tmp.append(w)
                    
            new_list.append(tmp)
            
        self.outputCorpusTokenized=new_list
        
            
    def prepare(self,):
        '''
           This is the final method in data preprocessing stage
           where corpus and keywords will be created for using
           in next stage e.g. Taxogen, LDA, HLDA etc. algorithms.
        '''  
        
        inpDataFileName=self.getFileNameFromPath(self.inputFile)
        pickleDataFileName='preprocess_'+inpDataFileName+'.pkl'
        
        inpKeyFileName=self.getFileNameFromPath(self.inputKeyFile)
        pickleKeyFileName='preprocess_'+inpKeyFileName+'.pkl'
        
        
        if os.path.exists(os.path.join(self.tempPath,pickleDataFileName)):
            reviewSeries = pd.read_pickle(os.path.join(self.tempPath,pickleDataFileName))
            
            if os.path.exists(os.path.join(self.tempPath,pickleKeyFileName)):
                
                with open(os.path.join(self.tempPath,pickleKeyFileName),'rb') as fin:
                    self.outputKeys=pickle.load(fin)
            else:
                self.processKeyWords()  # TODO : Keyword generation when keyword file does not exist.
                
        else:
            self.processData()
            self.processKeyWords()
            reviewSeries=pd.Series(self.outputData)
            for key in self.outputKeys:
                lst=key.strip().split('_')
                if len(lst)>1:
                    d=self.subs(key)
                    reviewSeries.replace(to_replace=d["left"], value=d["right"],regex=True,inplace=True)
            
            reviewSeries.to_pickle(os.path.join(self.tempPath,pickleDataFileName))
            
        if self.volume != 100.0:
            v=self.volume/100
            upperLimit=reviewSeries.count()
            lowerLimit=int(upperLimit*v)
            index=np.random.choice(upperLimit, lowerLimit,replace=False)
            modReviewSeries=reviewSeries.loc[ index ]
            self.outputProcessedCorpus=list(modReviewSeries)
            self.wordTokenization(self.outputProcessedCorpus)
            
            if self.dataSetName in ('dblp','bbc'):
                with open(os.path.join(self.tempPath,self.dataSetName+'_target_label.pkl'),'rb') as fin:
                    label=np.array(pickle.load(fin))
                
                with open(os.path.join(self.tempPath,self.dataSetName+'_target_label_modified.pkl'),'wb') as fout:
                    pickle.dump(list(label),fout)
            
        else:
            self.outputProcessedCorpus=list(reviewSeries)
            self.wordTokenization(self.outputProcessedCorpus)
            if self.dataSetName in ('dblp','bbc'):
                with open(os.path.join(self.tempPath,self.dataSetName+'_target_label.pkl'),'rb') as fin:
                    label=np.array(pickle.load(fin))
                    
                with open(os.path.join(self.tempPath,self.dataSetName+'_target_label_modified.pkl'),'wb') as fout:
                    pickle.dump(list(label),fout)
        
        self.__processGroundTruth()
        print('[Info]: Data preprocess and cleaning is complete')