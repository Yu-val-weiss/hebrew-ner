from utils import ner, yap
from conf import DEV
from utils.yap_graph import prune_lattices
from conf import ENV, TEST

def make_dev():
    tok = ner.read_file_to_sentences_df(DEV.TOK)
    
    multi = ner.read_file_to_sentences_df(DEV.MULTI)
    
    pred_multi = ner.read_file_to_sentences_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/hpc_eval_results/tok_multi_cnn.txt')
        
    tok_str = ner.raw_toks_str_from_ner_df(tok)
    
    print("Gold morph")

    ma = yap.yap_ma_api(tok_str)
    
    pruned = prune_lattices(ma, multi)
    
    md = yap.yap_joint_from_lattice_api(pruned)
    
    md['TOKEN'] = md['TOKEN'].astype(str)
    
    with open('utils_eval_files/yap_hybrid_gold_multi_dev_tokens.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['TOKEN'].agg('\n'.join)))
    
    md['FORM'] = md['FORM'].apply(lambda x: x + ' O')

    with open('utils_eval_files/yap_hybrid_gold_multi_dev.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['FORM'].agg('\n'.join)))

    print("Pred morph")
    
    pruned = prune_lattices(ma, pred_multi)
    
    md = yap.yap_joint_from_lattice_api(pruned)
    
    md['TOKEN'] = md['TOKEN'].astype(str)
    
    with open('utils_eval_files/yap_hybrid_pred_multi_dev_tokens.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['TOKEN'].agg('\n'.join)))
    
    md['FORM'] = md['FORM'].apply(lambda x: x + ' O')

    with open('utils_eval_files/yap_hybrid_pred_multi_dev.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['FORM'].agg('\n'.join)))

def make_test():
    tok = ner.read_file_to_sentences_df(TEST.TOK)
    
    multi = ner.read_file_to_sentences_df(TEST.MULTI)
    
    pred_multi = ner.read_file_to_sentences_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/ncrf_results/tok-multi/final/test-results.txt')
        
    tok_str = ner.raw_toks_str_from_ner_df(tok)
    
    print("Gold morph")

    ma = yap.yap_ma_api(tok_str)
    
    pruned = prune_lattices(ma, multi)
    
    md = yap.yap_joint_from_lattice_api(pruned)
    
    md['TOKEN'] = md['TOKEN'].astype(str)
    
    with open('utils_eval_files/yap_hybrid_gold_multi_test_tokens.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['TOKEN'].agg('\n'.join)))
    
    md['FORM'] = md['FORM'].apply(lambda x: x + ' O')

    with open('utils_eval_files/yap_hybrid_gold_multi_test.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['FORM'].agg('\n'.join)))

    print("Pred morph")
    
    pruned = prune_lattices(ma, pred_multi)
    
    md = yap.yap_joint_from_lattice_api(pruned)
    
    md['TOKEN'] = md['TOKEN'].astype(str)
    
    with open('utils_eval_files/yap_hybrid_pred_multi_test_tokens.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['TOKEN'].agg('\n'.join)))
    
    md['FORM'] = md['FORM'].apply(lambda x: x + ' O')

    with open('utils_eval_files/yap_hybrid_pred_multi_test.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['FORM'].agg('\n'.join)))

if __name__ == '__main__':
    make_test()