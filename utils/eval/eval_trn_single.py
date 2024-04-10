from utils import ner
from utils.eval import consts
from utils.metric import get_ner_BMES

def eval_single_dev():
    print('Single to single')
    pred_single = ner.read_file_to_sentences_df('ncrf_results/transformer/token_single/results.txt')
    single = ner.read_file_to_sentences_df(consts.DEV.TOK)
    # for t, m in (zip(get_ner_BMES(pred_single['Label'].to_list()), map(lambda x: x.split('@')[1] + x.split('@')[0], ner.make_spans(pred_single['Label'].to_list())))):
    #     if t != m:
    #         print(t, m)
    return ner.evaluate_token_ner(pred_single['Label'].to_list(), single['Label'].to_list())

def eval_single_test():
    print('Single to single')
    pred_single = ner.read_file_to_sentences_df('ncrf_results/transformer/token_single/test-results.txt')
    single = ner.read_file_to_sentences_df(consts.TEST.TOK)
    # for t, m in (zip(get_ner_BMES(pred_single['Label'].to_list()), map(lambda x: x.split('@')[1] + x.split('@')[0], ner.make_spans(pred_single['Label'].to_list())))):
    #     if t != m:
    #         print(t, m)
    return ner.evaluate_token_ner(pred_single['Label'].to_list(), single['Label'].to_list())

if __name__ == '__main__':
    eval_single_dev()