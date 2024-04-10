from utils import ner
from utils.eval import consts

def eval_single_tok():
    print('Single to single')
    pred_single = ner.read_file_to_sentences_df('hpc_eval_results/wiki_single_results.txt')
    single = ner.read_file_to_sentences_df(consts.TOK)
    return ner.evaluate_token_ner(pred_single['Label'].to_list(), single['Label'].to_list())

if __name__ == '__main__':
    eval_single_tok()