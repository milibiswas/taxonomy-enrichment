#!/bin/bash

#
#    Parameters:
#                $1 => Name of the input path (papers.txt, keywords.txt)
#                $2 => Name of the output path (where taxonomy and others will be created)
#                $3 => Name of the corpus (e.g. DBLP, AMAZON_REVIEW, NEWS_GROUP_20, BBC etc)
#                $4 => Clustering Information
#

## Name of the input directory from where three input files will be read

inputPath=${1}
export inputPath

## Name of the output directory where taxonomy will be stored along with other details
outputPath=${2}
export outputPath

## Name of the input corpus

corpusName=${3}
export corpusName

clusterInfo=${4}
export clusterInfo

maxLevel=${5}
export maxLevel

if [ -d "$inputPath" ]; then
# Take action if $inputPath exists. #
    echo "Trying to read files from ${inputPath}..."

    if [ -f "${inputPath}/papers.txt" ]; then
        echo "${inputPath}/papers.txt exist"
    else
        echo "${inputPath}/papers.txt does not exist, program exiting"
        exit 1

    fi

    if [ -f "${inputPath}/embeddings.txt" ]; then
        echo "${inputPath}/embeddings.txt exist"
    else
        echo "${inputPath}/embeddings.txt does not exist, program exiting"
        exit 1
    fi


    if [ -f "${inputPath}/keywords.txt" ]; then
        echo "${inputPath}/keywords.txt exist"
    else
        echo "${inputPath}/keywords.txt does not exist, program exiting"
        exit 1
    fi

fi

if [ -d "$outputPath" ]; then
    # Take action if $outputPath exists. #
    echo "Output dierctory ${outputPath} already exists!"

else
    echo "${outputPath} does not exists... creating the path"
    mkdir ${outputPath}

fi


if [ -z ${corpusName} ]
then
     echo "<corpusName> can't be empty! Exiting"
     exit 1
fi


## Name of the taxonomy
taxonName=our-l3-0.25
## If need preprocessing from raw input, set it to be 1, otherwise, set 0
FIRST_RUN=${FIRST_RUN:- 0}

if [ $FIRST_RUN -eq 0 ]; then
	echo 'Start data preprocessing'
	## compile word2vec for embedding learning
    #gcc word2vec.c -o word2vec -lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result

	## create initial folder if not exist

    if [ ! -d "${outputPath}/${corpusName}/raw" ]; then
        echo "Path ${outputPath}/${corpusName}/raw does not exist.. creating"
        mkdir -p ${outputPath}/${corpusName}/raw
    else
        rm -Rf ${outputPath}/${corpusName}/raw/*
    fi

    cp ${inputPath}/embeddings.txt ${outputPath}/${corpusName}/raw/embeddings.txt
    cp ${inputPath}/keywords.txt ${outputPath}/${corpusName}/raw/keywords.txt
    cp ${inputPath}/papers.txt ${outputPath}/${corpusName}/raw/papers.txt


    if [ ! -d "${outputPath}/${corpusName}/input" ]; then
        mkdir -p ${outputPath}/${corpusName}/input
    else
        rm -Rf ${outputPath}/${corpusName}/input/*
    fi

    cp ${inputPath}/embeddings.txt ${outputPath}/${corpusName}/input/embeddings.txt
    cp ${inputPath}/keywords.txt ${outputPath}/${corpusName}/input/keywords.txt
    cp ${inputPath}/papers.txt ${outputPath}/${corpusName}/input/papers.txt


    if [ ! -d "${outputPath}/${corpusName}/init" ]; then
		mkdir -p ${outputPath}/${corpusName}/init
    else
        rm -Rf ${outputPath}/${corpusName}/init/*
	fi

	echo 'Start cluster-preprocess.py'
	time python cluster-preprocess.py $corpusName

	echo 'Start preprocess.py'
	time python preprocess.py $corpusName

	cp ${outputPath}/${corpusName}/input/embeddings.txt ${outputPath}/${corpusName}/init/embeddings.txt
	cp ${outputPath}/${corpusName}/input/keywords.txt ${outputPath}/${corpusName}/init/seed_keywords.txt
fi

## create root folder for taxonomy
if [ ! -d ${outputPath}/${corpusName}/${taxonName} ]; then
	mkdir -p ${outputPath}/${corpusName}/${taxonName}
fi

## remove our-l3-0.15  folder
if [ -d "${outputPath}/${corpusName}/our-l3-0.15" ]; then
    rm -Rf ${outputPath}/${corpusName}/our-l3-0.15/*
fi

echo '[Info]: Running TaxonGen algorithm'
python3 main.py ${outputPath}/${corpusName}/ "${clusterInfo}" ${maxLevel}

echo 'Generate compressed taxonomy'
if [ ! -d ${outputPath}/${corpusName}/taxonomies ]; then
	mkdir -p ${outputPath}/${corpusName}/taxonomies
fi
#python3 compress.py -root ${outputPath}/${corpusName}/${taxonName} -output ${outputPath}/${corpusName}/taxonomies/${taxonName}.txt
