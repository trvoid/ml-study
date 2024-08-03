################################################################################
# TextVectorization 기초                                                       #
################################################################################

import os, sys, traceback, argparse
from datetime import datetime
from tensorflow.keras.layers import TextVectorization
import pickle

################################################################################
# Functions                                                                    #
################################################################################

def get_current_datetime_str():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def build_text_vectorization(text_data):
    # TextVectorization 층 생성
    text_vectorization = TextVectorization(
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=10
    )

    # 텍스트 데이터 적용하여 사전 생성
    text_vectorization.adapt(text_data)

    return text_vectorization

def save_text_vectorization(text_vectorization, output_filepath):
    with open(output_filepath, 'wb') as f:
       pickle.dump(text_vectorization, f)

def load_text_vectorization(filepath):
    with open(filepath, 'rb') as f:
        text_vectorization = pickle.load(f)

    return text_vectorization

def test_text_vectorization(text_vectorization):
    # 생성된 사전 출력
    vocabulary = text_vectorization.get_vocabulary()
    print('\n>>> Print vocabulary.')
    print(vocabulary)

    # 정수 인덱스를 단어로 변환
    integer_sequence = [0, 1, 2, 3, 4]  # 예시 정수 시퀀스
    words = [vocabulary[index] for index in integer_sequence]
    print('\n>>> Convert integer sequence to word sequence')
    print('> Integer sequence')
    print(integer_sequence)
    print('> Word sequence')
    print(words)

    # 텍스트 데이터를 정수 시퀀스로 변환
    sequence = text_vectorization(text_data)
    print('\n>>> Convert text data to sequences.')
    print(text_data)
    print('> Sequences (Tensor)')
    print(sequence)
    print('> Sequences (NumPy)')
    print(sequence.numpy())

################################################################################
# Configuration                                                                #
################################################################################

OUTPUT_FILENAME_PREFIX = 'text_vectorization'
OUTPUT_FILENAME_EXTENSION = '.pkl'

# 텍스트 데이터 생성 (예시)
text_data = ["This is a sample sentence", "Another example sentence"]

################################################################################
# Main                                                                         #
################################################################################

def main(args):
    output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        print(f'Directory not found: {output_dir}')
        return

    print('\n===== BUILD, TEST =================================================')
    text_vectorization = build_text_vectorization(text_data)
    test_text_vectorization(text_vectorization)

    print('\n===== SAVE ========================================================')
    output_filename = f'{OUTPUT_FILENAME_PREFIX}_{get_current_datetime_str()}{OUTPUT_FILENAME_EXTENSION}'
    output_filepath = os.path.join(output_dir, output_filename)

    save_text_vectorization(text_vectorization, output_filepath)
    print(f'*** OUTPUT_FILEPATH: {output_filepath}')

    print('\n===== LOAD, TEST ==================================================')
    text_vectorization = load_text_vectorization(output_filepath)
    test_text_vectorization(text_vectorization)
    
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='TextVectorization')
        parser.add_argument(
            '--output-dir', 
            required=True,
            help='Output directory.'
        )
        args = parser.parse_args()

        main(args)
    except:
        traceback.print_exc(file=sys.stdout)
