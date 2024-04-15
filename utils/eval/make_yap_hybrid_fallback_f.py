from utils import ner, yap
from config import DEV
from utils.yap_graph import prune_lattices
from config import ENV

if __name__ == '__main__':
    tok = ner.read_file_to_sentences_df(DEV.TOK)
    
    multi = ner.read_file_to_sentences_df(DEV.MULTI)
    
    pred_multi = ner.read_file_to_sentences_df(f'{ENV.ABSOLUTE_PATH_HEBREW_NER}/hpc_eval_results/tok_multi_cnn.txt')
        
    tok_str = ner.raw_toks_str_from_ner_df(tok)
    
    ma = yap.yap_ma_api(tok_str)
    
    print("Gold morph w/ fallback")
    
    pruned = prune_lattices(ma, multi, fallback=True)
    
    md = yap.yap_joint_from_lattice_api(pruned)
    
    md['TOKEN'] = md['TOKEN'].astype(str)
    
    with open('utils_eval_files/yap_hybrid_gold_multi_fallback_dev_tokens.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['TOKEN'].agg('\n'.join)))
    
    md['FORM'] = md['FORM'].apply(lambda x: x + ' O')

    with open('utils_eval_files/yap_hybrid_gold_multi_fallback_dev.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['FORM'].agg('\n'.join)))

    print("Pred morph w/ fallback")
    
    pruned = prune_lattices(ma, pred_multi, fallback=True)
    
    md = yap.yap_joint_from_lattice_api(pruned)
    
    md['TOKEN'] = md['TOKEN'].astype(str)
    
    with open('utils_eval_files/yap_hybrid_pred_multi_fallback_dev_tokens.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['TOKEN'].agg('\n'.join)))
    
    md['FORM'] = md['FORM'].apply(lambda x: x + ' O')

    with open('utils_eval_files/yap_hybrid_pred_multi_fallback_dev.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['FORM'].agg('\n'.join)))