################################################################################
# SentencePiece Test
################################################################################

import os, sys, traceback, argparse
import psutil, time
import sentencepiece as spm
import pandas as pd
import csv

################################################################################
# Functions
################################################################################

# 메모리 사용량 출력
def print_memory_usage(message):
    print(f'*** MEMORY USAGE: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.1f} MB ({message}) ***')

# 경과 시간 출력
def print_elapsed_time(start_time, end_time, message=''):
    elapsed_time = end_time - start_time
    message = f' ({message})' if len(message) > 0 else ''
    print(f'*** ELAPSED TIME: {elapsed_time:0.2f} seconds{message} ***')

# 파일 경로 출력
def print_output_filepath(filepath, message=''):
    message = f' ({message})' if len(message) > 0 else ''
    print(f'*** OUTPUT FILE{message}: {filepath}')

################################################################################
# Variables
################################################################################

files_all = [
    'data/won01-buljoyogyeong.txt',
    'data/won02-daesanjongsabeobeo.txt',
    'data/won03-jeongsanjongsabeobeo.txt',
    'data/won04-gyojeon.txt',
    'data/won05-gyosa.txt',
    'data/won06-yejeon.txt',
    'data/wiki01-won.txt',
    'data/wiki02-park.txt',
    'data/wiki03-wonkwang-univ.txt',
    'data/namu01-won.txt',
    'data/news01-ohmy.txt',
    'data/dic01-won.txt',
    'data/wonnews01-pyouh.txt'
]

files_one = [
    'data/won04-gyojeon.txt'
]

max_sentence_length = 9999

################################################################################
# Main
################################################################################

def main(args):
    alg = args.alg
    files = args.files

    data_file = ','.join(files_all) if files == 'all' else ','.join(files_one)

    output_dir = f'./output/sentencepiece/{alg}_{files}'
    os.makedirs(output_dir, exist_ok=True)

    model_prefix = os.path.join(output_dir, 'tokenizer')

    vocab_size = 8352 if alg == 'unigram' and files == 'all' else 24707 

    start_time = time.time()

    spm.SentencePieceTrainer.Train(f'--input={data_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={alg} --max_sentence_length={max_sentence_length} --minloglevel=1')

    print_elapsed_time(start_time, time.time())
    print_memory_usage('processing done')
    print_output_filepath(model_prefix)

    vocab_list = pd.read_csv(f'{model_prefix}.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
    print(f'훈련 결과 단어 목록 크기: {len(vocab_list)}')
    print(vocab_list.sample(10))

    sp = spm.SentencePieceProcessor()
    vocab_file = f"{model_prefix}.model"
    sp.load(vocab_file)

    lines = [
        "개교표어",
        "19.대종사 말씀하시기를 [스승이 법을 새로 내는 일이나, 제자들이 그 법을 받아서 후래 대중에게 전하는 일이나, 또 후래 대중이 그 법을 반가이 받들어 실행하는 일이 삼위 일체(三位一體)되는 일이라, 그 공덕도 또한 다름이 없나니라.]",
        "원불교는 대종사께서 창시하셨고 일원상 진리를 가르치신다."
    ]
    for line in lines:
        print('---------------------------------------')
        print(line)
        print()
        print(sp.encode_as_pieces(line))
        print()
        print(sp.encode_as_ids(line))
        print()

        print(sp.encode('개교표어', out_type=str))
        print(sp.encode('개교표어', out_type=int))

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description="SentencePiece Test"
        )
        parser.add_argument(
            "--alg",
            type=str,
            choices=['bpe', 'unigram'],
            required=True,
            help="Specifies the algorithm.",
        )
        parser.add_argument(
            "--files",
            type=str,
            choices=['all', 'one'],
            required=True,
            help="Specifies the option on the selection of training files.",
        )
        args = parser.parse_args()

        main(args)
    except:
        traceback.print_exc(file=sys.stdout)
