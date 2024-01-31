from utils import ner
from utils.eval import consts


if __name__ == '__main__':

    pred_multi = ner.read_file_to_sentences_df('hpc_eval_results/tok_multi_cnn.txt')
    
    print('Multi to single')
    
    single = ner.read_file_to_sentences_df(consts.TOK)

    ner.evaluate_token_ner(pred_multi['Label'].to_list(), single['Label'].to_list(), multi_tok=True)