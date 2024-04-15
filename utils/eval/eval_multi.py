import config
from utils import ner


def eval_multi_dev():
    pred_multi = ner.read_file_to_sentences_df('hpc_eval_results/tok_multi_cnn.txt')
    
    print('Multi to single')
    
    single = ner.read_file_to_sentences_df(config.DEV.TOK)

    return ner.evaluate_token_ner(pred_multi['Label'].to_list(), single['Label'].to_list(), multi_tok=True)

def eval_multi_test():
    pred_multi = ner.read_file_to_sentences_df('ncrf_results/tok-multi/final/test-results.txt')
    print('Multi to single')   
    single = ner.read_file_to_sentences_df(config.TEST.TOK)

    return ner.evaluate_token_ner(pred_multi['Label'].to_list(), single['Label'].to_list(), multi_tok=True)

if __name__ == '__main__':
    eval_multi_dev()