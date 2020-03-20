# taxonomy-build-algorithm
This repository contains different algorithms that are used to build taxonomy from different text corpus.

## Dataset used :

1. Amazon Fashion Review
2. BBC news data
3. DBLP network data

## Algorithms used:

1. Taxogen : Unsupervised way to build taxonomy
2. HLDA : Hierarchical Latent Dirichlet Allocation
3. Taxogen with no local embedding
4. Taxogen with no adaptive clustering
5. HCLUS : Hierarchical Clustering
6. Nethiex : Taxonomy building based on chinese resturant process

           


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

1. First install the required packages by running setup.py

2. Once the installation is compplete, run the below command to start the lgorithm


### python run.py -d < Name of the dataset > -n < Amount of data to be processed, values between 0 and 1 > -a < Name of the algorithm >

Example:

### python run.py -d 'amazon_fashion' -n 0.001 -a "taxogen"


## Visualize (Hypertree)

Check the file  in 

### ./src/code/hypertree/Visualisation/Fashion_vis.html



## Note:

HLDA & Nethiex integration are under development.<br/>
BBC, DBLP dataset integration are under development.
