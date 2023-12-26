import re
from typing import Callable, Iterable, List, NamedTuple, Tuple
from utils.metric import get_ner_BMES, get_ner_fmeasure, fmeasure_from_file

class WordLabel(NamedTuple):
    word: str
    label: str

def read_file(file: str, comment_delim='#', label_category_delim=' '):
    with open(file, encoding="utf-8") as f:
        return [
            WordLabel(*l.strip().split(label_category_delim))
            for l in f
            if l.strip() and not l.startswith(comment_delim)
        ]
        
def read_file_to_sentences(file: str, comment_delim='#', label_category_delim=' ') -> List[List[WordLabel]]:
    sent = []
    curr_sent = []
    with open(file, encoding="utf-8") as f:
        for l in f:
            if l.startswith(comment_delim):
                continue
            l = l.strip()
            if l:
                curr_sent.append(WordLabel(*l.split(label_category_delim)))
            else:
                sent.append(curr_sent)
                curr_sent = []
                
    return sent
    
# based on Appendix A in paper
def validate_multi_to_single(tag: str, multi_delim='^'):
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
        
    else:
        # In case the sequence of labels is not valid (e.g., B comes after E, or there is an O between two I labels),
        # we use a relaxed mapping that does not take the order of the labels into consideration
        # Fig. 11
        if 'S' in biose_str or ('B' in biose_str and 'E' in biose_str):
            single = 'S'
        elif 'E' in biose_str:
            single = 'E'
        elif 'B' in biose_str:
            single = 'B'
        elif 'I' in biose_str:
            single = 'I'
        else:
            single = 'O'

    if single != 'O':
            single += f"-{first_cat}"
            
    return single, valid


def make_spans(labels: Iterable[str]) -> List[str]:
    """Returns label spans from labels for the purposes of evaluation. 

    Args:
        labels (list[str]): list of labels

    Returns:
        list of tuple ((int, int), string): ((low, high), category)
    """
    spans: List[str] = []
    for i, label in enumerate(labels):
        if label == 'O' or 'I' in label:
            continue
        pos, cat = label.split('-')
        if pos == 'S' or pos == 'B':
            spans.append(f'{cat}@{i}')
        elif pos == 'E':
            spans[-1] += f',{i}'
    return spans


def evaluate_token_ner(pred: List[str], gold: List[str], multi_tok=False, multi_delim='^', beta=1):
    '''
    (Tjong Kim Sang and De Meulder 2003) for evaluation.
    
    Precision is the percentage of named entities found by the learning system that are correct.
    
    Recall is the percentage of named entities present in the corpus that are found by the system.
    
    A named entity is correct only if it is an exact match of the corresponding entity in the data file.
    
    Paper defaults beta for f_beta score to 1
    '''
    
    # we can ignore BIOSE labels, only interested in category and location
    
    
    assert len(pred) == len(gold)
    print("Evaluating performance...")
    
    corr_toks = sum(p == g for p,g in zip(pred, gold))
    
    print(f"Correct tokens: {corr_toks}, Total tokens: {len(pred)}, Accuracy: {corr_toks / len(pred):.4f}")
    if not multi_tok:
        pred_span, gold_span = make_spans(pred), make_spans(gold)
    else:
        lam: Callable[[str], str] = lambda x: validate_multi_to_single(x, multi_delim)[0]
        pred_span, gold_span = make_spans(map(lam, pred)), make_spans(map(lam, gold))
    
    
    
    # from above:
    # Precision is the percentage of named entities found by the learning system that are correct.
    # Recall is the percentage of named entities present in the corpus that are found by the system.
    correct = len(set(pred_span).intersection(set(gold_span)))
    precision = correct / len(pred_span)
    recall = correct / len(gold_span)
    f_beta = ((beta**2 + 1) * precision * recall) / (beta**2 * precision + recall)
    
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F{beta}: {f_beta:.4f}")
    
    
if __name__ == '__main__':
    res = [x.label for x in read_file("hpc_eval_results/tok_multi.txt")]
    gold = [x.label for x in read_file("/Users/yuval/GitHub/NEMO-Corpus/data/spmrl/gold/token-multi_gold_dev.bmes")]
    
    