import utils.ner as ner
import utils.yap as yap
from utils.eval.consts import TOK

if __name__ == '__main__':
    tok = ner.read_file_to_sentences_df(TOK)
        
    tok_str = ner.raw_toks_str_from_ner_df(tok)

    _, md = yap.yap_joint_api(tok_str)
    
    md['FORM'] = md['FORM'].apply(lambda x: x + ' O')


    with open('utils_eval_files/yap_morph_dev.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['FORM'].agg('\n'.join)))

