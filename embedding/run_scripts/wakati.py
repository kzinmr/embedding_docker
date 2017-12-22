import argparse
import MeCab
import neologdn
"""
USAGE:
python wakati.py -i '/tmp/wikipedia_raw_20170601_20160502.txt' -o '/tmp/wikipedia_wakati_20170601_20160502.txt'
"""

class WordDivider:

    def __init__(self, dictionary="mecabrc", is_normalize=False):
        self.dictionary = dictionary
        self.tagger = MeCab.Tagger(self.dictionary)
        self.tagger.parse("") # necessary in Python3
        self.is_normalize = is_normalize

    def extract_words(self, text):
        if not text:
            return []

        words = []
        # normalize text before MeCab processing
        if self.is_normalize:
            text = neologdn.normalize(text)
            text = text.lower()

        node = self.tagger.parseToNode(text)
        while node:
            features = node.feature.split(',')
            # extract word surface (configurable)
            word_to_append = node.surface
            words.append(word_to_append)
            node = node.next

        return words

def make_wakati_from_corpus(filepath, outpath):
    wd = WordDivider(is_normalize=True)
    docs = []
    with open(filepath, encoding='utf-8', errors='ignore') as ifp, open(outpath, 'w', encoding='utf-8') as ofp:
        for line in ifp:
            words = wd.extract_words(line)
            line_to_write = ' '.join(words)
            ofp.write(line_to_write + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    args = parser.parse_args()
    inputpath = args.input
    outputpath = args.output
    make_wakati_from_corpus(inputpath, outputpath)

if __name__ == "__main__":
    main()
