import re
from typing import NamedTuple

class WordLabel(NamedTuple):
    word: str
    label: str

def read_file(file, comment_delim='#', label_category_delim=' '):
    with open(file, encoding="utf-8") as f:
        return [
            WordLabel(*l.strip().split(label_category_delim))
            for l in f
            if l.strip() and not l.startswith(comment_delim)
        ]
    
# based on Appendix A in paper
def validate_multi_to_single(tag, multi_delim='^'):
    valid_seq = re.compile(r'O+|O*BI*(EO*)?|I+|I*EO*|O*SO*')
    biose_seq, cat_seq = zip(*[('O', None) if '-' not in label else label.split('-') for label in tag.split(multi_delim)])
    
    first_cat = next((cat for cat in cat_seq if cat is not None), '')
    
    valid = valid_seq.match(biose_str := ''.join(biose_seq)) is not None
    
    single = ''
    
    # now fix into a single label
    if valid:
        # pick regex expressions for each one, use anchors ^ and $ to exact match
        BIOSE = {
            'B': re.compile(r'^O*BI*$'), # only contains O, B and I
            'I': re.compile(r'^I+$'), # only contains I
            'O': re.compile(r'^O+$'), # only contains O
            'S': re.compile(r'^O*(S|BI*E)O*$'), # if either S, or contains both B and E (with optional I)
            'E': re.compile(r'^I*EO*$') # only contains optional I, E and optional O 
        }
        for lab in BIOSE:
            if BIOSE[lab].match(biose_str):
                single = lab
                break
        if single != 'O':
            cat_seq[0]
                
    else:
        pass
    
if __name__ == '__main__':
    ls = read_file("hpc_eval_results/tok_multi.txt")
    a, x = zip(*ls)
    print(list(ls))
    print(a)
    print(x)