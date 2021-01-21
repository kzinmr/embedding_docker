import argparse
import os
import random
import unicodedata
import MeCab
from typing import Optional


class MecabTokenizer:
    def __init__(self):
        self.tagger = MeCab.Tagger("-Owakati")

    def tokenize(self, text: str) -> Optional[str]:
        t = self.tagger.parse(text)
        if t is not None:
            return t.rstrip()
        else:
            return None


def preprocess(
    text: str, min_sentence_length: int = 10, min_line_length: int = 5
) -> str:
    blocks = []
    for block in text.split("\n\n"):
        if block.strip() and len(block) > min_sentence_length:
            t = "".join([l for l in block.split("\n") if len(l) > min_line_length])
            t = unicodedata.normalize("NFKC", t)
            t = t.lower()
            blocks.append(t)
    random.shuffle(blocks)
    return "".join(blocks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-corpus-dir",
        default="/app/workspace/raw",
        help="Location of raw text files.",
    )
    parser.add_argument(
        "--corpus-dir",
        default="/app/workspace/data",
        help="Location of pre-training text files.",
    )
    parser.add_argument(
        "--num-lines-per-file",
        default=10000,
    )
    args = parser.parse_args()

    tokenizer = MecabTokenizer()

    doc_dir = args.raw_corpus_dir
    n_docs = sum(
        [
            sum([1 for fn in files if fn.endswith(".txt")])
            for _, _, files in os.walk(doc_dir)
        ]
    )
    n_lines_per_file = args.num_lines_per_file
    docs_to_write = []
    n_current_docs = 0
    file_no = 0
    for cur, dirs, files in os.walk(doc_dir):
        doc_paths = [os.path.join(cur, fn) for fn in files if fn.endswith(".txt")]
        for doc_path in doc_paths:  # one doc per one file
            with open(doc_path) as fp:
                doc_text = preprocess(fp.read())
                doc_tokenized = tokenizer.tokenize(doc_text)
                if doc_tokenized is not None:
                    docs_to_write.append(doc_tokenized)
                n_current_docs += 1
            if n_current_docs % n_lines_per_file == 0 and docs_to_write:
                file_no += 1
                filepath = os.path.join(args.corpus_dir, f"corpus_{file_no}.txt")
                with open(filepath, "w", encoding='utf-8', errors='surrogateescape') as fp:
                    for l in docs_to_write:
                        fp.write(l)
                        fp.write("\n")
                docs_to_write = []
                n_current_docs = 0
    if docs_to_write:
        file_no += 1
        filepath = os.path.join(args.corpus_dir, f"corpus_{file_no}.txt")
        with open(filepath, "w", encoding='utf-8', errors='ignore') as fp:
            fp.write("\n\n".join(docs_to_write))
    print('Done preprocess')