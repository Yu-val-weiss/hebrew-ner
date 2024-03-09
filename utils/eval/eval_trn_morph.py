import pandas as pd
from utils.eval.consts import MORPH, MULTI, TOK
from utils import ner

from app_env import ENV

if __name__ == '__main__':
    PRED_MORPH = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/transformer/morph/results.txt'
    morph, pred_morph = ner.read_file_to_sentences_df(MORPH), ner.read_file_to_sentences_df(PRED_MORPH)
    tok, multi = ner.read_file_to_sentences_df(TOK), ner.read_file_to_sentences_df(MULTI)
    print('GOLD MORPH')
    ner.evaluate_morpheme(pred_morph, morph, multi, tok)
    
    
    print('\n\nYAP MORPH')
    
    tok_grouped = tok.groupby('SentNum')['Label'].agg(list).to_list()
    
    # print('PURE YAP')
    
    # ORIGINS = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_morph_dev_tokens.txt'
    # YAP_MORPH = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/hpc_eval_results/morph_cnn_seed_50_yap.txt'
    
    # yap_morph = ner.read_file_to_sentences_df(YAP_MORPH)
    # origins = ner.read_token_origins_to_df(ORIGINS)
    
    # merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True).groupby('SentNum')['Label'].agg(list).to_list()

    # ner.evaluate_token_ner_nested(merged, tok_grouped)
    
    print('\nHYBRID + PRED MULTI')
    
    yap_morph = ner.read_file_to_sentences_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/transformer/morph/pred_multi_results.txt')
    origins = ner.read_token_origins_to_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_hybrid_trn_pred_multi_dev_tokens.txt')
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True).groupby('SentNum')['Label'].agg(list).to_list()

    ner.evaluate_token_ner_nested(merged, tok_grouped)
    
    print('\nHYBRID + GOLD MULTI')
    
    yap_morph = ner.read_file_to_sentences_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/transformer/morph/gold_multi_results.txt')
    origins = ner.read_token_origins_to_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_hybrid_gold_multi_dev_tokens.txt')
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True)
    
    merged = merged.groupby('SentNum')['Label'].agg(list).to_list()

    ner.evaluate_token_ner_nested(merged, tok_grouped)