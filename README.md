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

- For execution, trigger the run python script from root folder

```
    $ python run.py [arguments]
```

- Command-line arguments for the program:

 | -a (Algorithm) | -d (Dataset) |  -n (Data Volume ratio to be used)
 | -------- | -------- | -------- |
 | taxogen    | amazon_fashion  | (0,100] |
 | ncrp       | bbc             |  |
 | taxogen_nole       | dblp            |  |
 | taxogen_noac       |      |  |
 | hclus       |                 |  |



### Execution examples

- Run algorithm (taxogen) on a dataset (amazon_fashion) using 10% of total data volume

```
    $ python run.py -d 'amazon_fashion' -a "taxogen" -n 10

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
