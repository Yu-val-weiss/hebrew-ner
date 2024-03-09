import pandas as pd
from utils.eval.consts import MORPH, MULTI, TOK
from utils import ner

from app_env import ENV

def eval_morph_ftam():
    PRED_MORPH = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/morph/from_hpc_seed46_ftam/results.txt'
    morph, pred_morph = ner.read_file_to_sentences_df(MORPH), ner.read_file_to_sentences_df(PRED_MORPH)
    tok, multi = ner.read_file_to_sentences_df(TOK), ner.read_file_to_sentences_df(MULTI)
    print('GOLD MORPH')
    gold_morph = ner.evaluate_morpheme(pred_morph, morph, multi, tok)
    
    
    print('\n\nYAP MORPH')
    
    tok_grouped = tok.groupby('SentNum')['Label'].agg(list).to_list()
    
    print('PURE YAP')
    
    ORIGINS = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_morph_dev_tokens.txt'
    YAP_MORPH = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/morph/from_hpc_seed46_ftam/yap_results.txt'
    
    yap_morph = ner.read_file_to_sentences_df(YAP_MORPH)
    origins = ner.read_token_origins_to_df(ORIGINS)
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True).groupby('SentNum')['Label'].agg(list).to_list()

    pure_yap = ner.evaluate_token_ner_nested(merged, tok_grouped)
    
    print('\nHYBRID + PRED MULTI')
    
    yap_morph = ner.read_file_to_sentences_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/morph/from_hpc_seed46_ftam/hybrid_pred_results.txt')
    origins = ner.read_token_origins_to_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_hybrid_pred_multi_dev_tokens.txt')
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True).groupby('SentNum')['Label'].agg(list).to_list()

    pred_multi = ner.evaluate_token_ner_nested(merged, tok_grouped)
    
    print('\nHYBRID + GOLD MULTI')
    
    yap_morph = ner.read_file_to_sentences_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/morph/from_hpc_seed46_ftam/hybrid_gold_results.txt')
    origins = ner.read_token_origins_to_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_hybrid_gold_multi_dev_tokens.txt')
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True)
    
    merged = merged.groupby('SentNum')['Label'].agg(list).to_list()

    gold_multi = ner.evaluate_token_ner_nested(merged, tok_grouped) 
    
    
    return (gold_morph[1], pure_yap, pred_multi, gold_multi)    # return only morph to single

if __name__ == '__main__':
    eval_morph_ftam()