import config
from utils import ner
from utils.eval.conf_interval import norm_approx_int
from utils.eval.eval_morph_ftam import DEV_SPANS, TEST_SPANS

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
    res = eval_single_test()
    
    print(" & ".join([
            f'{value*100:.2f} & {norm_approx_int(value, 0.95, TEST_SPANS)*100:.2f}'
            for metric, value in res._asdict().items()
        ]))