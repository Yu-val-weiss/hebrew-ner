from utils.eval.consts import MORPH, MULTI, TOK
from utils import ner

if __name__ == '__main__':
    PRED_MORPH = '/Users/yuval/GitHub/hebrew-ner/hpc_eval_results/morph_cnn_seed_50.txt'
    morph, pred_morph = ner.read_file_to_sentences_df(MORPH), ner.read_file_to_sentences_df(PRED_MORPH)
    tok, multi = ner.read_file_to_sentences_df(TOK), ner.read_file_to_sentences_df(MULTI)
    print('GOLD MORPH')
    ner.evaluate_morpheme(pred_morph, morph, multi, tok)
    
    
    print('\n\nYAP MORPH')
    
    print('PURE YAP')
    
    ORIGINS = '/Users/yuval/GitHub/hebrew-ner/utils_eval_files/yap_morph_dev_tokens.txt'
    YAP_MORPH = '/Users/yuval/GitHub/hebrew-ner/hpc_eval_results/morph_cnn_seed_50_yap.txt'
    
    yap_morph = ner.read_file_to_sentences_df(YAP_MORPH)
    origins = ner.read_token_origins_to_df(ORIGINS)
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True)

    ner.evaluate_token_ner(merged['Label'].to_list(), tok['Label'].to_list())
    
    print('\nHYBRID + PRED MULTI')
    
    yap_morph = ner.read_file_to_sentences_df('/Users/yuval/GitHub/hebrew-ner/hpc_eval_results/morph_cnn_seed_50_yap_hybrid_pred_multi.txt')
    origins = ner.read_token_origins_to_df('/Users/yuval/GitHub/hebrew-ner/utils_eval_files/yap_hybrid_pred_multi_dev_tokens.txt')
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True)

    ner.evaluate_token_ner(merged['Label'].to_list(), tok['Label'].to_list())
    
    print('\nHYBRID + GOLD MULTI')
    
    yap_morph = ner.read_file_to_sentences_df('/Users/yuval/GitHub/hebrew-ner/hpc_eval_results/morph_cnn_seed_50_yap_hybrid_gold_multi.txt')
    origins = ner.read_token_origins_to_df('/Users/yuval/GitHub/hebrew-ner/utils_eval_files/yap_hybrid_gold_multi_dev_tokens.txt')
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True)
    
    print(merged[(x := (merged['Label'] != tok['Label']))])
    
    print(tok[x])

    ner.evaluate_token_ner(merged['Label'].to_list(), tok['Label'].to_list())