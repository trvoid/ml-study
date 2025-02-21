################################################################################
# tiktoken Test
################################################################################

import os, sys, traceback, argparse
import psutil, time
import logging
import tiktoken

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

def get_logger(log_file):
    logging.basicConfig(level=logging.INFO,
                        format='')
    logger = logging.getLogger('tiktoken_test')
    #console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    #logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

################################################################################
# Variables
################################################################################

log_file = os.path.join('logs', 'tiktoken_test.log')

logger = get_logger(log_file)

################################################################################
# Main
################################################################################

def main():
    encoding = tiktoken.get_encoding("cl100k_base")

    lines = [
        "개교표어",
        "19.대종사 말씀하시기를 [스승이 법을 새로 내는 일이나, 제자들이 그 법을 받아서 후래 대중에게 전하는 일이나, 또 후래 대중이 그 법을 반가이 받들어 실행하는 일이 삼위 일체(三位一體)되는 일이라, 그 공덕도 또한 다름이 없나니라.]",
        "원불교는 대종사께서 창시하셨고 일원상 진리를 가르치신다."
    ]
    for line in lines:
        encoded = encoding.encode(line)
        decoded = [encoding.decode([token_id]) for token_id in encoded]

        logger.info('================================================================')
        logger.info(line)
        logger.info(encoded)
        logger.info(decoded)

if __name__ == '__main__':
    try:
        main()
    except:
        traceback.print_exc(file=sys.stdout)
