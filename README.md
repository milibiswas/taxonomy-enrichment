# taxonomy-build-algorithm
This repository contains different algorithms that are used to build taxonomy from different text corpus.

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
 | taxogen    | amazon_fashion  | (0,1] |
 | hlda       | bbc             |  |
 | ncrp       | dblp            |  |
 | taxogen_nole       | 20newsgroup     |  |
 | taxogen_noac       |                 |  |
 | hclus      |       |                 |  |


### Execution examples

- Run algorithm (taxogen) on a dataset (amazon_fashion) using 10% of total data volume

```
    $ python run.py -d 'amazon_fashion' -a "taxogen" -n 0.1

```

## Visualize (Hypertree)

- Check the file the following path 

```
    ./src/code/hypertree/Visualisation/Fashion_vis.html
```

### Dataset
```
    Amazon Fashion Review
    BBC news
    DBLP network
    20Newsgroup
```
### Algorithms
```
    taxogen      : Unsupervised way to build taxonomy
    hlda         : Hierarchical Latent Dirichlet Allocation
    taxogen_nole : Taxogen with no local embedding
    taxogen_noac : Taxogen with no adaptive clustering
    hclus        : Hierarchical Clustering
    ncrp         : Taxonomy building based on chinese resturant process
```
## Note:

HLDA is under development.<br/>
20Newsgroup dataset addition is under development
