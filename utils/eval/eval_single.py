import config
from utils import ner

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
    eval_single_dev()