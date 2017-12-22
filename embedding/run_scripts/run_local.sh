#!/bin/bash
# ./run_local.sh 20171203 20171215 5 10 256


WIKIPEDIA_VERSION=$1
NEOLOGD_VERSION=$2
window=${3-5}
min_count=${4-10}
dimension=${5-256}

#workspace=/workspace

wikipedia_raw=./wikipedia_raw_${WIKIPEDIA_VERSION}_${NEOLOGD_VERSION}.txt
wikipedia_wakati=./wikipedia_wakati_${WIKIPEDIA_VERSION}_${NEOLOGD_VERSION}.txt

# break down sentences in wikipedia corpus into words
python ./wakati.py -i $wikipedia_raw -o $wikipedia_wakati

# learn word2vec model from the corpus
w2v_model=./w2vmodel_wikipedia_${WIKIPEDIA_VERSION}_${NEOLOGD_VERSION}.model
python ./make_word2vec.py -i $wikipedia_wakati -o $w2v_model --min_count $min_count --window $window --dimension $dimension
