import config
from utils import ner
from utils.eval.conf_interval import norm_approx_int
from utils.eval.eval_morph_ftam import DEV_SPANS, TEST_SPANS
from utils.metric import get_ner_BMES

def eval_single_dev():
    print('Single to single')
    pred_single = ner.read_file_to_sentences_df('ncrf_results/transformer/token_single/results.txt')
    single = ner.read_file_to_sentences_df(config.DEV.TOK)
    # for t, m in (zip(get_ner_BMES(pred_single['Label'].to_list()), map(lambda x: x.split('@')[1] + x.split('@')[0], ner.make_spans(pred_single['Label'].to_list())))):
    #     if t != m:
    #         print(t, m)
    return ner.evaluate_token_ner(pred_single['Label'].to_list(), single['Label'].to_list())

def eval_single_test():
    print('Single to single')
    pred_single = ner.read_file_to_sentences_df('ncrf_results/transformer/token_single/test-results.txt')
    single = ner.read_file_to_sentences_df(config.TEST.TOK)
    # for t, m in (zip(get_ner_BMES(pred_single['Label'].to_list()), map(lambda x: x.split('@')[1] + x.split('@')[0], ner.make_spans(pred_single['Label'].to_list())))):
    #     if t != m:
    #         print(t, m)
    return ner.evaluate_token_ner(pred_single['Label'].to_list(), single['Label'].to_list())

if __name__ == '__main__':
    res = eval_single_test()
    
    print(" & ".join([
            f'{value*100:.2f} & {norm_approx_int(value, 0.95, TEST_SPANS)*100:.2f}'
            for metric, value in res._asdict().items()
        ]))