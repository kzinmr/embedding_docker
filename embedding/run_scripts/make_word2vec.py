import argparse
import multiprocessing
from gensim.models import word2vec


"""
USAGE:
python make_word2vec.py -i /tmp/corpus_wikipedia_wakati_20170601_20160502.txt -o /tmp/w2vmodel_wikipedia_20170601_20160502.model

HOW TO LOAD MODEL:
mloaded = word2vec.KeyedVectors.load('/tmp/w2vmodel_wikipedia_20170601_20160502.model')
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    parser.add_argument('--dimension', '-d', type=int, default=250)
    parser.add_argument('--window', '-w', type=int, default=5)
    parser.add_argument('--min_count', type=int, default=40)
    parser.add_argument('--workers', type=int, default=-1)
    parser.add_argument('--sg', type=int, default=1)
    args = parser.parse_args()
    inputpath = args.input
    outputpath = args.output
    sentences = word2vec.LineSentence(inputpath)
    # training
    mc = multiprocessing.cpu_count()
    workers = mc if args.workers == -1 else args.workers
    model = word2vec.Word2Vec(sentences,
                              size=args.dimension,
                              window=args.window,
                              min_count=args.min_count,
                              workers=workers,
                              sg=args.sg)
    # not saving temporary data
    model.delete_temporary_training_data()
    model.save(outputpath)


if __name__ == "__main__":
    main()
