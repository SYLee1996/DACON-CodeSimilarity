import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import logging, AutoTokenizer, RobertaTokenizer
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi
from itertools import combinations

logging.set_verbosity_error()

import warnings
warnings.filterwarnings(action='ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Inference', add_help=False)

    # Model parameters
    parser.add_argument('--code_folder', default='/home/CODE_SIMILARITY/code', type=str)
    parser.add_argument('--model_name', default='microsoft/codebert-base', type=str)
    parser.add_argument('--test_size', default=0.1, type=float)

    return parser


def main(args):
    
    # ------------------------------------------------------------------------------------------
    config = {
        'code_folder': args.code_folder,
        'model_name': args.model_name,
        'test_size': args.test_size,
        }
    
    seed = 10
    problem_folders = os.listdir(config['code_folder'])
    problem_folders.sort()

    # ------------------------------------------------------------------------------------------
    def preprocess_script(script):
        '''
        간단한 전처리 함수
        주석 -> 삭제
        '    '-> tab 변환
        다중 개행 -> 한 번으로 변환
        '''
        with open(script,'r',encoding='utf-8') as file:
            lines = file.readlines()
            preproc_lines = []
            for line in lines:
                if line.lstrip().startswith('#'):
                    continue
                line = line.rstrip()
                if '#' in line:
                    if re.search(r"'(.+?)'", line) != None:
                        continue
                    elif re.search(r'"(.+?)"', line) != None:
                        continue
                    else:
                        line = line[:line.index('#')]
                line = line.replace('\n','')
                line = line.replace('    ','\t')
                if line == '':
                    continue
                preproc_lines.append(line)
            preprocessed_script = '\n'.join(preproc_lines)
        return preprocessed_script

    preproc_scripts = []
    problem_nums = []

    for problem_folder in tqdm(problem_folders):
        scripts = os.listdir(os.path.join(config['code_folder'],problem_folder))
        problem_num = scripts[0].split('_')[0]
        for script in scripts:
            script_file = os.path.join(config['code_folder'],problem_folder,script)
            preprocessed_script = preprocess_script(script_file)

            preproc_scripts.append(preprocessed_script)
        problem_nums.extend([problem_num]*len(scripts))
        
    # ------------------------------------------------------------------------------------------
    df = pd.DataFrame(data = {'code':preproc_scripts, 'problem_num':problem_nums})

    # ------------------------------------------------------------------------------------------
    if (config['model_name'] == 'cross-encoder/ms-marco-MiniLM-L-12-v2') | \
        (config['model_name'] == "cross-encoder/ms-marco-electra-base") | \
        (config['model_name'] == "huggingface/CodeBERTa-small-v1") | \
        (config['model_name'] == "sentence-transformers/paraphrase-xlm-r-multilingual-v1"):
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)
    else :
        tokenizer = RobertaTokenizer.from_pretrained(config['model_name'], use_fast=True)
            
    df['tokens'] = df['code'].apply(tokenizer.tokenize)
    df['len'] = df['tokens'].apply(len)

    # ------------------------------------------------------------------------------------------
    ndf = df[df['len'] <= 1000].reset_index(drop=True)

    # ------------------------------------------------------------------------------------------
    train_df, valid_df, train_label, valid_label = train_test_split(
            ndf,
            ndf['problem_num'],
            random_state=seed,
            test_size=config['test_size'],
            stratify=ndf['problem_num'],
        )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    # ------------------------------------------------------------------------------------------
    for df_set, name in zip([train_df, valid_df], ['train', 'valid']):

        codes = df_set['code'].to_list()
        problems = df_set['problem_num'].unique().tolist()
        problems.sort()

        tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
        bm25 = BM25Okapi(tokenized_corpus)

        total_positive_pairs = []
        total_negative_pairs = []

        for problem in tqdm(problems):
            solution_codes = df_set[df_set['problem_num'] == problem]['code']
            positive_pairs = list(combinations(solution_codes.to_list(),2)) # 조합 생성
            
            solution_codes_indices = solution_codes.index.to_list()
            negative_pairs = []
            
            first_tokenized_code = tokenizer.tokenize(positive_pairs[0][0])
            negative_code_scores = bm25.get_scores(first_tokenized_code)
            negative_code_ranking = negative_code_scores.argsort()[::-1] # 내림차순
            ranking_idx = 0

            for solution_code in solution_codes:
                negative_solutions = []
                while len(negative_solutions) < len(positive_pairs) // len(solution_codes):
                    high_score_idx = negative_code_ranking[ranking_idx]
                    
                    if high_score_idx not in solution_codes_indices:
                        negative_solutions.append(df_set['code'].iloc[high_score_idx])
                    ranking_idx += 1

                for negative_solution in negative_solutions:
                    negative_pairs.append((solution_code, negative_solution))
            total_negative_pairs.extend(negative_pairs)
            total_positive_pairs.extend(positive_pairs)

        pos_code1 = list(map(lambda x:x[0],total_positive_pairs))
        pos_code2 = list(map(lambda x:x[1],total_positive_pairs))

        neg_code1 = list(map(lambda x:x[0],total_negative_pairs))
        neg_code2 = list(map(lambda x:x[1],total_negative_pairs))

        pos_label = [1]*len(pos_code1)
        neg_label = [0]*len(neg_code1)

        pos_code1.extend(neg_code1)
        total_code1 = pos_code1
        pos_code2.extend(neg_code2)
        total_code2 = pos_code2
        pos_label.extend(neg_label)
        total_label = pos_label
        pair_data = pd.DataFrame(data={
            'code1':total_code1,
            'code2':total_code2,
            'similar':total_label
        })
        pair_data = pair_data.sample(frac=1).reset_index(drop=True)

        pair_data.to_csv('{}_data.csv'.format(name +"_"+ config['model_name'].split("/")[-1] ),index=False)
        print('{}_data.csv is saverd!'.format(name +"_"+ config['model_name'].split("/")[-1] ))
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser('test script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)