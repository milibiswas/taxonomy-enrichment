#####################################################
#
#    file name : generate_taxonomy.py
#    created by: Mili Biswas (MSc - Comp. Sc, UNIFR)
#    creation Dt: 28th Nov 2019
#
#    
#
#####################################################

import os


import create_network as cn
import eval_modified as ev
import build_cluster as bc
import get_top_words as topw
import create_json as jsn


wrk_dir = './data/'

if os.path.exists(os.path.join(wrk_dir,"db.elist.ncrp")):
    os.remove(os.path.join(wrk_dir,"db.elist.ncrp"))
if os.path.exists(os.path.join(wrk_dir,"dblp_cs.pkl")):
    os.remove(os.path.join(wrk_dir,"dblp_cs.pkl"))
    
#cn.main_title()
ev.embedding(wrk_dir, ev.NetData(wrk_dir, 'word_cooc'), ev.AlgoNCRP, 'db')
bc.cluster()
topw.main()
#jsn.main()


