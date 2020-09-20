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
#from lxml import etree as ElementTree
from contextlib import redirect_stdout
import sys
#from zipfile import ZipFile
import os
#import gzip
from collections import deque
import re
#import collections
#import pickle
#import urllib.request
#import xml.etree.ElementTree

import pandas as pd
#import numpy as np
#from sklearn.datasets import load_files

import nltk
from nltk.stem import WordNetLemmatizer
#from nltk import  FreqDist

#from preprocess import keyword_extract as ke

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

class CustomErrorKeywordFileMissing(Error):
    pass

#---------------------------------
# Class that preprocess the corpus
#---------------------------------

class Data(object):
    
    def __init__(self,inputFile,inputKeyFile):
        
        # Input Parameters
        self.inputFile=inputFile
        self.inputKeyFile=inputKeyFile
        self.tempPath='./src/data/tmp/'
        
        # Output Parameters
        
        self.outputData=None
        self.outputKeys=None
        self.outputCorpusTokenized=None
        self.outputProcessedCorpus=None
        
        # Other Parameters
        self.lem=WordNetLemmatizer()
        self.stopWord = self.buildStopWords(self.readStopWords('./src/data/input/other/en_stopwords.txt'))
        #self.stopWord = self.buildStopWords(en_stop)
        self.most_commonn_word = None

        
    
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

    def processKeyWords(self,):
        '''
           processKeyWords() method preprocess keywords data.
           Each keyword (where mutliple words exist) is being
           joined by underscore ('_').
        '''
        try:   
            if self.fileExist(self.inputKeyFile):
                self.__setKeywordData(self.inputKeyFile)
            else:
                raise CustomErrorKeywordFileMissing
        except CustomErrorKeywordFileMissing:
            print('[Error]: {} file is missing'.format(self.inputKeyFile))
            sys.exit(1)
        except Exception as e:
            print('[Error]: '+str(e))
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
                return None
            else:
                raise Exception
        except  Exception as e:
            print('[Error]: Preprocessing data. {}'.format(str(e)))
            sys.exit(1)
    
    def wordTokenization(self,corpus):
        final_corpus = [self.removeStopWords(review) for review in corpus if review.strip() !='']
        word_tokenized_corpus = [review.split() for review in final_corpus]
        self.outputCorpusTokenized=word_tokenized_corpus

    def prepare(self,):
            self.processData()
            self.processKeyWords()
            reviewSeries=pd.Series(self.outputData)
            for key in self.outputKeys:
                lst=key.strip().split('_')
                if len(lst)>1:
                    d=self.subs(key)
                    reviewSeries.replace(to_replace=d["left"], value=d["right"],regex=True,inplace=True)
            self.outputProcessedCorpus=list(reviewSeries)
            self.wordTokenization(self.outputProcessedCorpus)
        