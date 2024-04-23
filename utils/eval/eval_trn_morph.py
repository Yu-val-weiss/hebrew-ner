import pandas as pd
from config import DEV
from utils import ner

from config import ENV, TEST
from utils.eval.conf_interval import norm_approx_int
from utils.eval.eval_morph_ftam import DEV_SPANS, TEST_SPANS

def eval_trn_morph_dev():
    PRED_MORPH = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/transformer/morph/results.txt'
    morph, pred_morph = ner.read_file_to_sentences_df(DEV.MORPH), ner.read_file_to_sentences_df(PRED_MORPH)
    tok, multi = ner.read_file_to_sentences_df(DEV.TOK), ner.read_file_to_sentences_df(DEV.MULTI)
    print('GOLD MORPH')
    return ner.evaluate_morpheme(pred_morph, morph, multi, tok)[1]
    
    
def eval_trn_morph_test():
    PRED_MORPH = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/transformer/morph/test-results.txt'
    morph, pred_morph = ner.read_file_to_sentences_df(TEST.MORPH), ner.read_file_to_sentences_df(PRED_MORPH)
    tok, multi = ner.read_file_to_sentences_df(TEST.TOK), ner.read_file_to_sentences_df(TEST.MULTI)
    print('GOLD MORPH')
    return ner.evaluate_morpheme(pred_morph, morph, multi, tok)[1]

def eval_all_trn_morph_dev():
    PRED_MORPH = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/transformer/morph/results.txt'
    morph, pred_morph = ner.read_file_to_sentences_df(DEV.MORPH), ner.read_file_to_sentences_df(PRED_MORPH)
    tok, multi = ner.read_file_to_sentences_df(DEV.TOK), ner.read_file_to_sentences_df(DEV.MULTI)
    print('GOLD MORPH')
    gold_morph_to_morph, gold_morph = ner.evaluate_morpheme(pred_morph, morph, multi, tok)
    
    
    print('\n\nYAP MORPH')
    
    tok_grouped = tok.groupby('SentNum')['Label'].agg(list).to_list()
    
    print('PURE YAP')
    
    ORIGINS = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_morph_dev_tokens.txt'
    YAP_MORPH = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/transformer/morph/yap_morph_results.txt'
    
    yap_morph = ner.read_file_to_sentences_df(YAP_MORPH)
    origins = ner.read_token_origins_to_df(ORIGINS)
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True).groupby('SentNum')['Label'].agg(list).to_list()

    pure_yap = ner.evaluate_token_ner_nested(merged, tok_grouped)
    
    print('\nHYBRID + PRED MULTI')
    
    yap_morph = ner.read_file_to_sentences_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/transformer/morph/pred_multi_results.txt')
    origins = ner.read_token_origins_to_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_hybrid_trn_pred_multi_dev_tokens.txt')
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True).groupby('SentNum')['Label'].agg(list).to_list()

    pred_multi = ner.evaluate_token_ner_nested(merged, tok_grouped)
    
    print('\nHYBRID + GOLD MULTI')
    
    yap_morph = ner.read_file_to_sentences_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/transformer/morph/gold_multi_results.txt')
    origins = ner.read_token_origins_to_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_hybrid_gold_multi_dev_tokens.txt')
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True)
    
    merged = merged.groupby('SentNum')['Label'].agg(list).to_list()

    gold_multi = ner.evaluate_token_ner_nested(merged, tok_grouped)
    
    return gold_morph_to_morph, gold_morph, pure_yap, pred_multi, gold_multi

def eval_all_trn_morph_test():
    PRED_MORPH = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/transformer/morph/test-results.txt'
    morph, pred_morph = ner.read_file_to_sentences_df(TEST.MORPH), ner.read_file_to_sentences_df(PRED_MORPH)
    tok, multi = ner.read_file_to_sentences_df(TEST.TOK), ner.read_file_to_sentences_df(TEST.MULTI)
    print('GOLD MORPH')
    gold_morph_to_morph, gold_morph = ner.evaluate_morpheme(pred_morph, morph, multi, tok)
    
    
    print('\n\nYAP MORPH')
    
    tok_grouped = tok.groupby('SentNum')['Label'].agg(list).to_list()
    
    print('PURE YAP')
    
    ORIGINS = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_morph_test_tokens.txt'
    YAP_MORPH = f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/transformer/morph/test-yap_morph_results.txt'
    
    yap_morph = ner.read_file_to_sentences_df(YAP_MORPH)
    origins = ner.read_token_origins_to_df(ORIGINS)
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True).groupby('SentNum')['Label'].agg(list).to_list()

    pure_yap = ner.evaluate_token_ner_nested(merged, tok_grouped)
    
    print('\nHYBRID + PRED MULTI')
    
    yap_morph = ner.read_file_to_sentences_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/transformer/morph/test-pred_multi_results.txt')
    origins = ner.read_token_origins_to_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_hybrid_trn_pred_multi_test_tokens.txt')
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True).groupby('SentNum')['Label'].agg(list).to_list()

    pred_multi = ner.evaluate_token_ner_nested(merged, tok_grouped)
    
    print('\nHYBRID + GOLD MULTI')
    
    yap_morph = ner.read_file_to_sentences_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/transformer/morph/test-gold_multi_results.txt')
    origins = ner.read_token_origins_to_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/utils_eval_files/yap_hybrid_gold_multi_test_tokens.txt')
    
    merged = ner.merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True)
    
    merged = merged.groupby('SentNum')['Label'].agg(list).to_list()

    gold_multi = ner.evaluate_token_ner_nested(merged, tok_grouped)
    
    return gold_morph_to_morph, gold_morph, pure_yap, pred_multi, gold_multi

if __name__ == '__main__':
    res = eval_all_trn_morph_test()
    print('\n\n')
    for r, t in zip(res, ['gold morph-morph', 'gold morph-tok', 'pure yap', 'pred multi', 'gold multi']):
        print(t)
        print(" & ".join([
            f'{value*100:.2f} & {norm_approx_int(value, 0.95, TEST_SPANS)*100:.2f}'
            for metric, value in r._asdict().items()
        ]))
        print()