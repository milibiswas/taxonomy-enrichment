#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =======================================================================================================
# 
#       Name : build_cluster.py
#       Description: build json (Taxonomy Tree) using nethiex algorithm
#       Created by : Mili Biswas
#       Created on: 21.02.2020
#
#       Dependency : Must be called from generate_taxonomy.py
#
# ========================================================================================================

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import json
import sys
  
class cluster():
    
    def __init__(self,dirPath,clusterInfo,maxDepth,keywordFile,jsonPath):
        self.dirPath=dirPath
        self.clusterInfo=clusterInfo
        self.maxDepth=maxDepth
        self.keywordFile=keywordFile
        self.jsonPath=jsonPath
        self.dataDict=None
        self.id2wd=None
        
        
    def __setBuildId2Word(self,n):
        id2wd = ['-1' for _ in range(n)]
        with open(self.dirPath + 'db.voc',encoding='iso-8859-1') as fin:
            cnt = 0
            for line in fin:
                cnt += 1
                i, wd = line.split()
                i, wd = int(i), wd.strip()
                id2wd[i] = wd
        self.id2wd=id2wd
        
    def __getTopWord(self,centerVec,dataVec,id2wd): 
        dist = euclidean_distances(dataVec,centerVec)
        row,_=dist.shape
        dist_reshaped=np.reshape(dist,(row,))
        min_idx = np.argsort(dist_reshaped)[0]
        word=id2wd[min_idx]
        return word
    
    
    def __getId2Wd(self,dataPos,id2wd):
        id2wd_modified={}
        for key,val in enumerate(dataPos):
            id2wd_modified[key]=id2wd[val]
        
        return id2wd_modified
    
    def getDataValue(self,):
        return self.dataDict
    
    def __setDataValue(self,value):
        self.dataDict=value
        return None
    
    def __readKeywords(self,keywordFile):
        keywordVecPos=[]
        with open(keywordFile,'r') as fin:
            for line in fin:
                for key,value in enumerate(self.id2wd):
                    if value.upper()==line.strip('\n').upper():
                        keywordVecPos.append(key)
                        break
                    else:
                        continue
        return keywordVecPos
                                    
    def __createJavaScriptFile(self,path=None):
        try:
            if self.dataDict==None:
                raise Exception
            else:
                with open(path,'w') as fout:
                    fout.write('function json_data() {')
                    fout.write('\n')
                    fout.write('var x = ')
                    json.dump(self.dataDict,fout)
                    fout.write(';')
                    fout.write('return x; }')
                print('[Info]: Jason file created at :',path)
        except Exception as err:
            print('Error occurred')
            print(str(err))
            sys.exit(1)

    def __getCenterOfClustersInfo(self,inputData,numberOfCluster):
        '''
           This method returns centers of clusters as well as 
           indexes of each cluster datapoints
           
           Input Values:
               
               inputData : Numpy Array of vectors
               numberOfCluster : Number of Clusters that will be used in clustering algorithm e.g. kmeans
           
           Return value : retVal
           Data structure of retVal:
               [
                   {
                   
                     'cluster': <cluster number e.g. 0,1,2 etc>
                     'center' : < Center vector as list>
                     'data_position' : < Positions of data element of the cluster>
                   
                   }, ...
                   
               ]
           
        '''
        
        retVal=[]
        kmeans = KMeans(n_clusters=numberOfCluster, random_state=0).fit(inputData)
        clusterCenter=kmeans.cluster_centers_
        dataClusterLabels=kmeans.labels_
        
        for i_cat in range(numberOfCluster):
            tmp={}
            posList=[]
            tmp['cluster']=i_cat
            tmp['center']=list(clusterCenter[i_cat,:])
            for pos,elem in enumerate(list(dataClusterLabels)):
                if elem==i_cat:
                    posList.append(pos)
                    
            tmp['data_position']=posList
            
            retVal.append(tmp)
            
        return retVal
    
    def __buildCluster(self,inputData,d_node,id2wd,curDepth,maxDepth,root):
        '''
           This function generates the tree of taxonomy
        
        '''
        
        dataDict={}
        if root==1:
            root=0
            dataDict['id']='root'
            dataDict['name']='root'
            dataDict['queries']='false'
            dataDict['description']='Root Level'
        else:
            pass
            
        dataDict['children']=[]
        if len(self.clusterInfo)>0 and curDepth<maxDepth:
            firstVal=self.clusterInfo.pop(0)
            word_m=inputData[:, (d_node * curDepth):(d_node * (curDepth + 1))]
            ret=self.__getCenterOfClustersInfo(word_m,firstVal)
            for i in ret:
                
                nextFeats=inputData[i['data_position']]
                vec=np.array(i['center'])
                center=np.reshape(vec,(1,-1))
                
                ##=================================
                # Calling the method __getTopWord()
                ##=================================
                
                nodeName=self.__getTopWord(center,word_m[i['data_position']],id2wd)
                
                nextId2Wd=self.__getId2Wd(i['data_position'],id2wd)
            
                x=self.__buildCluster(nextFeats,d_node,nextId2Wd,curDepth+1,maxDepth,root)
                x['id']=nodeName
                x['name']=nodeName
                #x['center']=list(vec)   TODO  This is not needed now .. but required while implement incremental logic
                x['data']={
                  "type": "concept",
                  "depth": curDepth+1
                }
                dataDict['children'].append(x)
                
        else:
            for i in id2wd.values():
                data={"type": "concept","depth": curDepth+1}
                dataDict['children'].append({'id':i,'name':i,'data':data,'children':[]})
        return dataDict
    
    def prepare(self,):
        '''
            This function is a warpper that calls build_cluster() method
            which eventually generates the json (Taxonomy Tree)
        '''
        
        with open(self.dirPath+'db.elist.ncrp') as fin:
            
            '''
                 n is the number of vocabulary or vectors
                 d is the dimension of vector
            
            '''
            
            n, d = fin.readline().split()
            n, d = int(n), int(d)
            
            self.__setBuildId2Word(n)
    
            d_node = d // (self.maxDepth + 1)
    
            feats = np.zeros((n, d), dtype=np.float64)
            for line in fin:
                line = line.split()
                assert len(line) == d + 1
                feats[int(line[0])] = [float(x) for x in line[1:]]
                
        keyWordVectors=feats[self.__readKeywords(self.keywordFile)]
        #print(keyWordVectors)
                        
        retVal=self.__buildCluster(keyWordVectors,d_node,self.id2wd,0,self.maxDepth,root=1)
        self.__setDataValue(retVal)
        self.__createJavaScriptFile(self.jsonPath)
        print('[Info]: Taxonomy building is complete using Chinese Restaurant Process')
        print(retVal)
        return None
        
    
    
if __name__ == "__main__":
    print('[Info]: This is build cluster module from nethiex algorithm')
