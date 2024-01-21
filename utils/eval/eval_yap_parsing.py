from utils import yap, ner
from utils.eval.consts import MORPH

if __name__ == '__main__':
    gold = ner.read_file_to_sentences_df(MORPH).groupby('SentNum')['Word'].agg(list)
    
    
    hybrid_gold_multi = ner.read_file_to_sentences_df('/Users/yuval/GitHub/hebrew-ner/utils_eval_files/yap_hybrid_gold_multi_dev.txt').groupby('SentNum')['Word'].agg(list)
    
    
    tot = 0
    corr = 0
    
    for gs, hs in zip(gold, hybrid_gold_multi):
        for g, h in zip(gs, hs):
            if g == h:
                corr += 1
            tot += 1
            
    print("Hybrid gold:", corr/tot)    
    
    
    hybrid_pred_multi = ner.read_file_to_sentences_df('/Users/yuval/GitHub/hebrew-ner/utils_eval_files/yap_hybrid_pred_multi_dev.txt').groupby('SentNum')['Word'].agg(list)
    
    
    tot = 0
    corr = 0
    
    for gs, hs in zip(gold, hybrid_pred_multi):
        for g, h in zip(gs, hs):
            if g == h:
                corr += 1
            tot += 1
            
    print("Hybrid pred:", corr/tot)  
    
    
    yap = ner.read_file_to_sentences_df('/Users/yuval/GitHub/hebrew-ner/utils_eval_files/yap_morph_dev.txt').groupby('SentNum')['Word'].agg(list)
    
    
    tot = 0
    corr = 0
    
    for gs, hs in zip(gold, yap):
        for g, h in zip(gs, hs):
            if g == h:
                corr += 1
            tot += 1
            
    print("Pure yap:", corr/tot)  