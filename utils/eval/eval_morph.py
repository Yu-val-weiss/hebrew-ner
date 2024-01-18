from utils.eval.consts import MORPH, MULTI, TOK
import utils.ner as ner

if __name__ == '__main__':
    PRED_MORPH = '/Users/yuval/GitHub/hebrew-ner/hpc_eval_results/morph_cnn_seed_50.txt'
    morph, pred_morph = ner.read_file_to_sentences_df(MORPH), ner.read_file_to_sentences_df(PRED_MORPH)
    tok, multi = ner.read_file_to_sentences_df(TOK), ner.read_file_to_sentences_df(MULTI)
    
    ner.evaluate_morpheme(pred_morph, morph, multi, tok)
    
    print('Multi to multi')

    pred_multi = ner.read_file_to_sentences_df('/Users/yuval/GitHub/hebrew-ner/hpc_eval_results/tok_multi_cnn.txt')
    ner.evaluate_token_ner(pred_multi['Label'].to_list(), multi['Label'].to_list(), multi_tok=True)
    # print(pred_morph['Label'].dtype)