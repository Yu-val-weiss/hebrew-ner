import config
from utils import ner
from utils.eval.conf_interval import norm_approx_int
from utils.eval.eval_morph_ftam import DEV_SPANS, TEST_SPANS


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
    res = eval_multi_test()
    
    print(" & ".join([
            f'{value*100:.2f} & {norm_approx_int(value, 0.95, TEST_SPANS)*100:.2f}'
            for metric, value in res._asdict().items()
        ]))