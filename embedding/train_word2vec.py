import argparse
import multiprocessing
from gensim.models import word2vec
import codecs

codecs.register_error('strict', codecs.lookup_error('surrogateescape'))

"""
USAGE:
python make_word2vec.py -i /tmp/corpus_wikipedia_wakati_20170601_20160502.txt -o /tmp/w2vmodel_wikipedia_20170601_20160502.model

HOW TO LOAD MODEL:
mloaded = word2vec.KeyedVectors.load('/tmp/w2vmodel_wikipedia_20170601_20160502.model')
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir",
        "-i",
        default="/app/workspace/data",
        help="Location of pre-training text files.",
    )
    parser.add_argument('--output', '-o')
    parser.add_argument('--dimension', '-d', type=int, default=256)
    parser.add_argument('--window', '-w', type=int, default=16)
    parser.add_argument('--min-count', type=int, default=10)
    parser.add_argument('--max-vocab-size', type=int, default=30000)
    parser.add_argument('--max-sentence-length', type=int, default=30000)
    parser.add_argument('--workers', type=int, default=-1)
    parser.add_argument('--sg', type=int, default=1)
    args = parser.parse_args()
    outputpath = args.output
    mc = multiprocessing.cpu_count() // 2
    workers = mc if args.workers == -1 else args.workers
    sentences = word2vec.PathLineSentences(args.corpus_dir, max_sentence_length=args.max_sentence_length)
    model = word2vec.Word2Vec(sentences,
                              size=args.dimension,
                              window=args.window,
                              min_count=args.min_count,
                              max_vocab_size=args.max_vocab_size,
                              workers=workers,
                              sg=args.sg)
    # not saving temporary data
    model.delete_temporary_training_data()
    model.save(outputpath)
    model.wv.save_word2vec_format(f'{outputpath}.txt')


if __name__ == "__main__":
    main()
