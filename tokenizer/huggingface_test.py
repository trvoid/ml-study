################################################################################
# Hugging Face Tokenizers Test
################################################################################

import os, sys, traceback, argparse
import psutil, time
import glob
from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer

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

vocab_size = 30000
limit_alphabet = 6000
min_frequency = 5

################################################################################
# Main
################################################################################

def main(args):
    dir = args.dir
    tok = args.tok
    
    if not os.path.isdir(dir):
        raise Exception('Not a directory: ' + dir)
    
    dir = os.path.abspath(dir)
    basename = os.path.basename(dir)
    
    # 훈련용 텍스트 파일 목록 만들기
    data_file = glob.glob(os.path.join(dir, '*.txt'))

    # 결과 디렉토리 만들기
    output_dir = f'./output/huggingface/{basename}_{tok}'
    os.makedirs(output_dir, exist_ok=True)

    # 토크나이저 생성
    tokenizer = BertWordPieceTokenizer(lowercase=False, strip_accents=False) if tok == 'bpe' else SentencePieceBPETokenizer() 
    
    # 토크나이저 훈련
    start_time = time.time()

    tokenizer.train(files=data_file,
                    vocab_size=vocab_size,
                    limit_alphabet=limit_alphabet,
                    min_frequency=min_frequency)
    
    tokenizer.save_model(output_dir)
    
    print_elapsed_time(start_time, time.time())
    print_memory_usage('processing done')
    print_output_filepath(output_dir)

    # 토크나이저 사용
    lines = [
      "개교표어",
      "19.대종사 말씀하시기를 [스승이 법을 새로 내는 일이나, 제자들이 그 법을 받아서 후래 대중에게 전하는 일이나, 또 후래 대중이 그 법을 반가이 받들어 실행하는 일이 삼위 일체(三位一體)되는 일이라, 그 공덕도 또한 다름이 없나니라.]",
      "원불교는 대종사께서 창시하셨고 일원상 진리를 가르치신다."
    ]
    for line in lines:
      tokenized = tokenizer.encode(line)
      print('================================================================')
      print(line)
      print()
      print(tokenized)
      print()
      print(tokenized.tokens)
      print(tokenized.ids)

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description="Hugging Face Tokenizers Test"
        )
        parser.add_argument(
            "--dir",
            type=str,
            required=True,
            help="Specifies a directory that has text files for training.",
        )
        parser.add_argument(
            "--tok",
            type=str,
            choices=['bert', 'sp'],
            required=True,
            help="Specifies the tokenizer, BertWordPieceTokenizer or SentencePieceBPETokenizer",
        )
        args = parser.parse_args()

        main(args)
    except:
        traceback.print_exc(file=sys.stdout)
