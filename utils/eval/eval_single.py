import math

from tqdm import tqdm
import config
from utils import ner
import scipy.stats as st
import numpy as np
import pandas as pd

def eval_single_dev():
    print('Single to single')
    pred_single = ner.read_file_to_sentences_df('hpc_eval_results/wiki_single_results.txt')
    single = ner.read_file_to_sentences_df(config.DEV.TOK)
    return ner.evaluate_token_ner(pred_single['Label'].to_list(), single['Label'].to_list())

def eval_single_test():
    print('Single to single')
    pred_single = ner.read_file_to_sentences_df('ncrf_results/tok-single/final/test-results.txt')
    single = ner.read_file_to_sentences_df(config.TEST.TOK)
    return ner.evaluate_token_ner(pred_single['Label'].to_list(), single['Label'].to_list())

if __name__ == '__main__':
    # eval_single_dev()
    pred_single = ner.read_file_to_sentences_df('ncrf_results/tok-single/final/test-results.txt')
    test = ner.read_file_to_sentences_df(config.TEST.TOK)
    wordN = len(test)
    SentN = max(test['SentNum'])
    conf = 0.95 
    z = st.norm.ppf((1 + conf) / 2.0)
    f1 = 0.7815
    conf_int_length = z * math.sqrt((f1 * (1 - f1)) / wordN)

    lower = f1 - conf_int_length
    upper = f1 + conf_int_length
    
    print("len:", conf_int_length)
    print("low, up:", lower, upper)
    
    print('bootstrap!')
    f1s = []
    merged = pd.merge(pred_single, test.drop('Word', axis='columns'), on=['SentNum', 'WordIndex'], how='left', suffixes=['', '_Gold'])
    merged = merged.groupby('SentNum').agg(list)
    for i in tqdm(range(500), desc='Strapping boots'):
        bootstrap = merged.sample(n=SentN, replace=True)
        f1 = ner.evaluate_token_ner(bootstrap['Label'].explode().to_list(), bootstrap['Label_Gold'].explode().to_list(), verbose=False).f * 100
        f1s.append(f1)
        
    lower, upper = np.percentile(f1s, [2.5, 97.5])
    print("len:", upper-lower)
    print("low, up:", lower, upper)
    print("mean", np.mean(f1s))