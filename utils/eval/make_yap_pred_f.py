from config import TEST
from utils import ner, yap
from config import DEV

def make_dev():
    tok = ner.read_file_to_sentences_df(DEV.TOK)
        
    tok_str = ner.raw_toks_str_from_ner_df(tok)

    _, md = yap.yap_joint_api(tok_str)

    md['TOKEN'] = md['TOKEN'].astype(str)

    with open('utils_eval_files/yap_morph_dev_tokens.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['TOKEN'].agg('\n'.join)))

    md['FORM'] = md['FORM'].apply(lambda x: x + ' O')

    with open('utils_eval_files/yap_morph_dev.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['FORM'].agg('\n'.join)))
        
def make_test():
    tok = ner.read_file_to_sentences_df(TEST.TOK)
        
    tok_str = ner.raw_toks_str_from_ner_df(tok)

    _, md = yap.yap_joint_api(tok_str)

    md['TOKEN'] = md['TOKEN'].astype(str)

    with open('utils_eval_files/yap_morph_test_tokens.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['TOKEN'].agg('\n'.join)))

    md['FORM'] = md['FORM'].apply(lambda x: x + ' O')

    with open('utils_eval_files/yap_morph_test.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['FORM'].agg('\n'.join)))

if __name__ == '__main__':
    make_test()

