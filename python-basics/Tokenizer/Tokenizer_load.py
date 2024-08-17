################################################################################
# Tokenizer dump/load 테스트                                                     # 
# * 입력:                                                                       #
# * 출력:                                                                       #
################################################################################

import os, sys, traceback
import argparse

from pickle import load

def main(args):
    input_filepath = args.input_filepath

    sample_tokenizer = load(open(input_filepath, 'rb'))

    print(sample_tokenizer.word_index)

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Tokenizer dump/load test")
        parser.add_argument(
            "--input-filepath",
            type=str,
            choices=None,
            required=True,
            help="Specifies the dump file path for Tokenizer.",
        )
        args = parser.parse_args()

        main(args)
    except:
        traceback.print_exc(file=sys.stdout)
