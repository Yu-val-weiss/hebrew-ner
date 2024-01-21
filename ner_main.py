import time
from typing import Set, Tuple
from utils import yap, ner
from utils.yap_graph import YapGraph
import pandas as pd

def prune_lattices(lattice_df: pd.DataFrame, multi_df: pd.DataFrame, multi_label_delim='^', fallback=False):
    splitting = ner.make_multi_splitting_df(multi_df, multi_label_delim)
    valid_edges: Set[Tuple[int, int, int, int]] = set()
    for ((sent_id, tok_id), sub_lattice),(_, split) in zip(lattice_df.groupby(['SENTNUM', 'TOKEN']), splitting.groupby(['SentNum', 'WordIndex'])):
        # tok_id in lattice starts from 1, so may need to offset
        g = YapGraph.from_df(sub_lattice)
        source, target = sub_lattice['FROM'].iat[0], sub_lattice['TO'].iat[-1]
        path_len = split['Splitting'].iat[0] + 1
        paths = list(g.get_all_paths(source, target, limit=path_len))
        pruned_paths = [p for p in paths if len(p) == path_len]
        if fallback and len(pruned_paths) == 0:
            pruned_paths = [p for p in paths if abs(len(p) - path_len) <= 1]
        if len(pruned_paths) > 0:
            paths = pruned_paths
        for p in paths:
            for f, t in zip(p[:-1], p[1:]):
                valid_edges.add((sent_id, tok_id, f, t))
                
    cols_to_filter = ['SENTNUM', 'TOKEN', 'FROM', 'TO']
    return lattice_df[lattice_df[cols_to_filter].apply(tuple, axis=1).isin(valid_edges)].reset_index(drop=True)
    

if __name__ == '__main__':
    MORPH = "/Users/yuval/GitHub/NEMO-Corpus/data/spmrl/gold/morph_gold_dev.bmes"
    MULTI = "/Users/yuval/GitHub/NEMO-Corpus/data/spmrl/gold/token-multi_gold_dev.bmes"
    TOK = "/Users/yuval/GitHub/NEMO-Corpus/data/spmrl/gold/token-single_gold_dev.bmes"
    
    PRED_MORPH = '/Users/yuval/GitHub/hebrew-ner/hpc_eval_results/morph_cnn_seed_50.txt'
    
    morph = ner.read_file_to_sentences_df(MORPH)
    pred_morph = ner.read_file_to_sentences_df(PRED_MORPH)
    multi = ner.read_file_to_sentences_df(MULTI)
    tok = ner.read_file_to_sentences_df(TOK)
    tok = tok[tok['SentNum'] < 1000]
    
    multi_test = ner.read_file_to_sentences_df('/Users/yuval/GitHub/hebrew-ner/scratch/test.txt')
    
    # evaluate_morpheme(PRED_MORPH, MORPH, MULTI)
    
    s = "עשרות אנשים מגעים מתאילנד לישראל כשהם נרשמים כמתנדבים , אך למעשה משמשים עובדים שכירים זולים .  "
    s2 = "עשרות אנשים מגעים מתאילנד לישראל כשהם נרשמים כמתנדבים , אך למעשה משמשים עובדים שכירים זולים .  גנו גידל דגן בגן .  "
    # ma, md =  yap.yap_joint_api(s)
    # ma_ma = yap.yap_ma_api(s)
    
    ma = yap.yap_ma_api(ner.raw_toks_str_from_ner_df(tok))
    
    print("MA complete")
    
    x = prune_lattices(ma, multi)
    
    print("Pruning complete")
    
    
    print("Gold morph + fallback")

    ma = yap.yap_ma_api(ner.raw_toks_str_from_ner_df(tok))
    
    print(ma)
    
    pruned = prune_lattices(ma, multi, fallback=True)
    
    md = yap.yap_joint_from_lattice_api(pruned)
    
    md['TOKEN'] = md['TOKEN'].astype(str)
    
    with open('utils_eval_files/yap_hybrid_gold_multi_fallback_dev_tokens.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['TOKEN'].agg('\n'.join)))
    
    md['FORM'] = md['FORM'].apply(lambda x: x + ' O')

    with open('utils_eval_files/yap_hybrid_gold_multi_fallback_dev.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['FORM'].agg('\n'.join)))
    
    # print("Disambiguating lattices...")
    
    # s = time.time()
    
    # print(yap.yap_joint_from_lattice_api(ma))
    
    # print("Unpruned took", time.time() - s)
    
    # print("Dismabiguating pruned lattice...")
    
    # now = time.time()
    
    # print(yap.yap_joint_from_lattice_api(x))
    
    # print("Pruned took", time.time() - now)
    
    # md = yap.yap_joint_from_lattice_api(x)
    
    # print(md)