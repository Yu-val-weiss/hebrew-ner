import pandas as pd
from config import DEV
from utils import ner
from config import ENV

def eval_morph():
    PRED_MORPH = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/hpc_eval_results/morph_cnn_seed_50.txt'
    morph, pred_morph = ner.read_file_to_sentences_df(DEV.MORPH), ner.read_file_to_sentences_df(PRED_MORPH)
    tok, multi = ner.read_file_to_sentences_df(DEV.TOK), ner.read_file_to_sentences_df(DEV.MULTI)
    print('GOLD MORPH')
    gold_morph = ner.evaluate_morpheme(pred_morph, morph, multi, tok)
    
    
    print('\n\nYAP MORPH')
    
    tok_grouped = tok.groupby('SentNum')['Label'].agg(list).to_list()
    
    print('PURE YAP')
    
    ORIGINS = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_morph_dev_tokens.txt'
    YAP_MORPH = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/hpc_eval_results/morph_cnn_seed_50_yap.txt'
    
    yap_morph = ner.read_file_to_sentences_df(YAP_MORPH)
    origins = ner.read_token_origins_to_df(ORIGINS)
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True).groupby('SentNum')['Label'].agg(list).to_list()

    pure_yap = ner.evaluate_token_ner_nested(merged, tok_grouped)
    
    print('\nHYBRID + PRED MULTI')
    
    yap_morph = ner.read_file_to_sentences_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/hpc_eval_results/morph_cnn_seed_50_yap_hybrid_pred_multi.txt')
    origins = ner.read_token_origins_to_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_hybrid_pred_multi_dev_tokens.txt')
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True).groupby('SentNum')['Label'].agg(list).to_list()

    hybrid_pred_multi = ner.evaluate_token_ner_nested(merged, tok_grouped)
    
    print('\nHYBRID + GOLD MULTI')
    
    yap_morph = ner.read_file_to_sentences_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/hpc_eval_results/morph_cnn_seed_50_yap_hybrid_gold_multi.txt')
    origins = ner.read_token_origins_to_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_hybrid_gold_multi_dev_tokens.txt')
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True)
    
    merged = merged.groupby('SentNum')['Label'].agg(list).to_list()

    hybrid_gold_multi = ner.evaluate_token_ner_nested(merged, tok_grouped)
    
    return gold_morph[1], pure_yap, hybrid_pred_multi, hybrid_gold_multi
    
if __name__ == '__main__':
    eval_morph()