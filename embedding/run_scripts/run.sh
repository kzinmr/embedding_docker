#!/bin/bash
# This script contains parallelizable sections by Spark for example.
# mount by `docker run -v $(pwd)/run_scripts:/workspace embedding /workspace/run.sh`


set -o allexport;source /WIKIPEDIA_VERSION;set +o allexport
set -o allexport;source /NEOLOGD_VERSION;set +o allexport

workspace=/workspace

wikipedia_dirname=wikipedia_${WIKIPEDIA_VERSION}
wikipedia_dir=${workspace}/${wikipedia_dirname}
wikipedia_raw=${workspace}/wikipedia_raw_${WIKIPEDIA_VERSION}_${NEOLOGD_VERSION}.txt
wikipedia_wakati=${workspace}/wikipedia_wakati_${WIKIPEDIA_VERSION}_${NEOLOGD_VERSION}.txt

# break down sentences in wikipedia corpus into words
python ${workspace}/wakati.py -i $wikipedia_raw -o $wikipedia_wakati

# learn word2vec model from the corpus
w2v_model=${workspace}/w2vmodel_wikipedia_${WIKIPEDIA_VERSION}_${NEOLOGD_VERSION}.model
python ${workspace}/make_word2vec.py -i $wikipedia_wakati -o $w2v_model --min_count 10 --window 5 --dimension 256
