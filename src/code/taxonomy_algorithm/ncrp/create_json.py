import os
import json
import pickle

class node(object):
    def __init__(self,dictVal,l,catg):
        self.value = dictVal[catg]
        self.listVal=l
        self.category=catg
        self.child = []
        self.dictVal=dictVal
    def __creat_child__(self,):
        for lv,child in self.listVal:
            if lv==self.category:
                for cnt,catg in enumerate(child):
                    ch = node(self.dictVal,self.listVal,catg)
                    ch.__creat_child__()
                    self.child.append(ch)
        #temp_l.append(ch)
    def tree_traverse(self,):
        #data.append(root.value)
        data={}
        data["value"]=self.value
        data["category"]=self.category
        data["child"]=[]
        if len(self.child)==0:
            return data
        else:
            for i in self.child:
                ret=i.tree_traverse()
                data["child"].append(ret)
            return data

def main(path_dir='./data/'):
    #======================================        
    l = [(0,[1,2,3]),(1,[4,5,6]),(2,[7,8,9]),(3,[10])]
    cat_file_obj=open(os.path.join(path_dir,"cat_file.dict"),"rb")
    dictVal=pickle.load(cat_file_obj)
    cat_file_obj.close()
    #======================================
    parent = node(dictVal,l,0)
    parent.__creat_child__()
    x=parent.tree_traverse()
    print(json.dumps(x, indent=4, sort_keys=True))
    with open(os.path.join(path_dir,"category.json"),"w") as fileObj:
        json.dump(x,fileObj,indent=4)
