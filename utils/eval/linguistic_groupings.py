from conf import ENV
from utils.ner import read_file_to_sentences_df, make_groupings_linguistically

MORPH = f"{ENV.CORPUS_DIR}/data/spmrl/gold/morph_gold_test.bmes"
MULTI = f"{ENV.CORPUS_DIR}/data/spmrl/gold/token-multi_gold_test.bmes"
TOK = f"{ENV.CORPUS_DIR}/data/spmrl/gold/token-single_gold_test.bmes"

if __name__ == '__main__':
    tok = read_file_to_sentences_df(TOK).groupby('SentNum')['Word'].agg(list)
    morph = read_file_to_sentences_df(MORPH).groupby('SentNum')['Word'].agg(list)
    multi = read_file_to_sentences_df(MULTI).groupby('SentNum')['Label'].agg(list)
    
    
    correct_sent = 0
    tot_sent = 0
    for t, mo, mul in zip(tok, morph, multi):
        groups, ling_tok = make_groupings_linguistically(mo)
        pred = (list(map(len, groups)))
        gold = (list(map(lambda x: len(x.split('^')),mul)))
        if pred == gold:
            correct_sent += 1
        tot_sent += 1
    
    print(correct_sent, tot_sent, correct_sent / tot_sent)