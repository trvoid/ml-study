################################################################################
# Tokenizer dump/load 테스트                                                     # 
# * 입력:                                                                       #
# * 출력:                                                                       #
################################################################################

import os, sys, traceback
import argparse

from tensorflow.keras.preprocessing.text import Tokenizer
from pickle import dump

sample_texts = [
    '산악용 자전거'
    , '자전거 전조등 후미등'
    , '온라인 상품권'
    , '온라인 상품 세트'
]

def main(args):
    output_filepath = args.output_filepath
    
    sample_tokenizer = Tokenizer()
    sample_tokenizer.fit_on_texts(sample_texts)

    print(sample_tokenizer.word_index)

    print(sample_tokenizer.texts_to_sequences(sample_texts))

    dump(sample_tokenizer, open(output_filepath, 'wb'))

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Tokenizer dump/load test")
        parser.add_argument(
            "--output-filepath",
            type=str,
            choices=None,
            required=True,
            help="Specifies the dump file path for Tokenizer.",
        )
        args = parser.parse_args()

        main(args)
    except:
        traceback.print_exc(file=sys.stdout)
