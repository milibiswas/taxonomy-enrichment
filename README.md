# taxonomy-enrichment
This repository contains different algorithms that are used to build taxonomy from different text corpus. It also contains novel approach to enrich taxonomy when new set of data is available from same domain of knowledge.

## Prerequisites and dependencies

- Python 3.0 or higher
- Clone this repository.
- All the additional dependencies will be installed using the build script.

___

## Build

- Install all the dependent packages and modules using the setup script located in the root folder:
```
    $ python setup.py
```

___

## Execution

- For full run taxonomy algorithms, trigger the run python script from root folder as below

```
    $ python run.py [arguments]
```

- Command-line arguments for the program:

 | -a (Algorithm) | -d (Dataset) |  -n (Data Volume percent to be used)
 | -------- | -------- | -------- |
 | taxogen    | amazon_fashion  | (0,100] |
 | ncrp       | bbc             |  |
 | taxogen_nole       | dblp            |  |
 | taxogen_noac       |      |  |
 | hclus       |                 |  |

- For taxonomy enhancement run, trigger the run_enhance python script from root folder as below

```
    $ python run_enhance.py [arguments]
```

- Command-line arguments for the program:

 |-d (Dataset) |  -n (Data Volume percent to be used)
 | -------- | -------- | -------- |
 | amazon_fashion  | (0,100] |
 | bbc             |  |
 | dblp            |  |
 

### Execution examples

- Run enriching taxonomy algorithm on a dataset (amazon_fashion) using 10% of total data volume

```
    $ python run_enhance.py -d 'amazon_fashion' -n 10

```

## Visualize (Hypertree)

- Check the file the following path 

```
    ./src/data/output/hypertree/Visualisation/Fashion_vis.html
```

### Dataset
```
    Amazon Fashion Review
    BBC news
    DBLP network
```
### Algorithms
```
    taxogen      : Unsupervised way to build taxonomy
    taxogen_nole : Taxogen with no local embedding
    taxogen_noac : Taxogen with no adaptive clustering
    hclus        : Hierarchical Clustering
    ncrp         : Taxonomy building based on chinese resturant process
```
## Reference:
@inproceedings{rehurek_lrec,
      title = {{Software Framework for Topic Modelling with Large Corpora}},
      author = {Radim {\v R}eh{\r u}{\v r}ek and Petr Sojka},
      booktitle = {{Proceedings of the LREC 2010 Workshop on New
           Challenges for NLP Frameworks}},
      pages = {45--50},
      year = 2010,
      month = May,
      day = 22,
      publisher = {ELRA},
      address = {Valletta, Malta},
      language={English}
}

@article{DBLP:journals/corr/abs-1812-09551,
  author    = {Chao Zhang and
               Fangbo Tao and
               Xiusi Chen and
               Jiaming Shen and
               Meng Jiang and
               Brian M. Sadler and
               Michelle Vanni and
               Jiawei Han},
  title     = {TaxoGen: Unsupervised Topic Taxonomy Construction by Adaptive Term
               Embedding and Clustering},
  journal   = {CoRR},
  volume    = {abs/1812.09551},
  year      = {2018},
  url       = {http://arxiv.org/abs/1812.09551},
  archivePrefix = {arXiv},
  eprint    = {1812.09551},
  timestamp = {Fri, 15 Feb 2019 12:58:25 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1812-09551.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{DBLP:conf/kdd/MaCW018,
  author={Jianxin Ma and Peng Cui and Xiao Wang and Wenwu Zhu},
  title={Hierarchical Taxonomy Aware Network Embedding},
  year={2018},
  cdate={1514764800000},
  pages={1920-1929},
  url={https://doi.org/10.1145/3219819.3220062},
  booktitle={KDD},
  crossref={conf/kdd/2018}
}

Amazon Fashion Review Dataset: https://snap.stanford.edu/data/web-Amazon.html
BBC News Article Dataset: http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
DBLP Citation Network Dataset: https://dblp.uni-trier.de/xml/
