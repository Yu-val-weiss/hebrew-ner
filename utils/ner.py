# -*- coding: utf-8 -*-
# @Author: Yuval Weiss
import re
import string
import time
from typing import Callable, Iterable, List, NamedTuple, Tuple

from utils import yap
import pandas as pd
from utils.eval.consts import MORPH, MULTI, TOK

class WordLabel(NamedTuple):
    word: str
    label: str
    def concat(self, wl: 'WordLabel', label_delim = '^'):
        return WordLabel(self.word + wl.word,
                         self.label + label_delim + wl.label)
        
    
class EvaluationMetrics(NamedTuple):
    precision: float
    recall: float
    f: float
    
    def __str__(self) -> str:
        return f"Metrics:\nP: {self.precision:.4f}\tR: {self.recall:.4f}\tF: {self.recall:.4f}"

def read_file(file: str, comment_delim='#', word_label_delim=' '):
    with open(file, encoding="utf-8") as f:
        return [
            WordLabel(*l.strip().split(word_label_delim))
            for l in f
            if l.strip() and not l.startswith(comment_delim)
        ]
        
def read_file_to_sentences(file: str, comment_delim='#', word_label_delim=' ') -> List[List[WordLabel]]:
    sent = []
    curr_sent = []
    with open(file, encoding="utf-8") as f:
        for l in f:
            if l.startswith(comment_delim):
                continue
            l = l.strip()
            if l:
                curr_sent.append(WordLabel(*l.split(word_label_delim)))
            else:
                sent.append(curr_sent)
                curr_sent = []
                
    return sent


def read_file_to_sentences_df(file: str, comment_delim='#', word_label_delim=' '):
    '''
    Reads a file of sentences, with each word NER labelled into a pandas dataframe
    
    Columns: `['SentNum', 'WordIndex', 'Word', 'Label']` 
    
    Can use `r.groupby('SentNum').agg(list)` to get them aggregated into sentences
    '''
    def sent_iter():
        curr_sent = 0
        word_ind = 0
        with open(file, encoding="utf-8") as f:
            for l in f:
                if l.startswith(comment_delim):
                    continue
                if l := l.strip():
                    word, label = l.rsplit(word_label_delim, maxsplit=1)
                    yield [curr_sent, word_ind, word, label]
                    word_ind += 1
                else:
                    curr_sent += 1
                    word_ind = 0
      
    columns = ['SentNum', 'WordIndex', 'Word', 'Label'] 
                
    return pd.DataFrame(
        data = sent_iter(),
        columns = columns
    )
    

def read_token_origins_to_df(file: str, comment_delim='#'):
    '''
    Reads a file of sentences, with each word NER labelled into a pandas dataframe
    
    Note: Yap assigns 1-based indices, so will subtract 1 here 
    
    Columns: `['SentNum', 'WordIndex' 'Origin']` 
    
    Can use `r.groupby('SentNum').agg(list)` to get them aggregated into sentences
    '''
    def sent_iter():
        curr_sent = 0
        word_ind = 0
        with open(file, encoding="utf-8") as f:
            for l in f:
                if l.startswith(comment_delim):
                    continue
                if l := l.strip():
                    yield [curr_sent, word_ind, int(l) - 1]
                    word_ind += 1
                else:
                    curr_sent += 1
                    word_ind = 0
      
    columns = ['SentNum', 'WordIndex', 'Origin'] 
                
    return pd.DataFrame(
        data = sent_iter(),
        columns = columns
    )

def merge_morph_from_multi_spliting(morph: pd.DataFrame, multi: pd.DataFrame, multi_label_delim='^', validate_to_single=False):
    '''
    Merges morphological-based predictions based on the splitting in the multi model. Validates to a single token if `validate_to_single` is `True`.
    '''
    morph_sents = morph.groupby('SentNum').agg(list)
    splitting = make_multi_splitting_df(multi, multi_label_delim)
    splitting = splitting.groupby('SentNum').agg(list)[['WordIndex', 'Splitting']]
    def merge_morphs():
        for (sent_num, words, labels), (word_nums, sent_splits) in zip(morph_sents[['Word', 'Label']].itertuples(), splitting.itertuples(index=False)):
            word, label = [], []
            for word_ind, split in zip(word_nums, sent_splits):
                for _ in range(split):
                    word.append(words.pop(0))
                    label.append(labels.pop(0))
                ret_label = multi_label_delim.join(label)
                if validate_to_single:
                    ret_label = validate_multi_to_single(ret_label)[0]
                yield [sent_num, word_ind, multi_label_delim.join(word), ret_label]
                word, label = [], []


    columns = ['SentNum', 'WordIndex', 'MergedWord', 'Label']
    
    return pd.DataFrame(
        data = merge_morphs(),
        columns = columns,
    )
    
def merge_morph_from_token_origins(morph: pd.DataFrame, origins: pd.DataFrame, multi_label_delim='^', validate_to_single=False):
    '''
    Merges morphological-based predictions based on the splitting from the yap request's token origins. Validates to a single token if `validate_to_single` is `True`.
    '''
    df = (pd
            .merge(morph, origins, on=['SentNum', 'WordIndex'])
            .drop('WordIndex', axis='columns')
            .groupby(['SentNum', 'Origin'])
            .agg(multi_label_delim.join)
            .reset_index()
            .rename(columns={'Origin': 'WordIndex'}))
    
    if validate_to_single:
        df['Label'] = df['Label'].apply(lambda x: validate_multi_to_single(x, multi_label_delim)[0])
        
    return df

def make_multi_splitting_df(multi: pd.DataFrame, multi_label_delim='^'):
    '''
    Columns: `['SentNum', 'WordIndex', 'Splitting']`
    '''
    splitting = pd.DataFrame(
        {
            'SentNum': multi['SentNum'],
            'WordIndex': multi['WordIndex'],
            'Splitting': multi['Label'].apply(lambda x: len(x.split(multi_label_delim)))
        }
    )
    
    return splitting


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



def normalise_final_letters(word: str):
    hebrew_mapping = {
        'ך': 'כ',
        'ם': 'מ',
        'ן': 'נ',
        'ף': 'פ',
        'ץ': 'צ'
    }

    # Replace final Hebrew letters with their normal forms
    for final_letter, normal_form in hebrew_mapping.items():
        word = word.replace(final_letter, normal_form)

    return word

def correct_final_letters(word: str):
    if len(word) < 2:
        return word
    hebrew_mapping = {
        'כ': 'ך',
        'מ': 'ם',
        'נ': 'ן',
        'פ': 'ף',
        'צ': 'ץ'
    }
    if word[-1] in hebrew_mapping:
        return word[:-1] + hebrew_mapping[word[-1]]
    return word

def remove_trailing_yud(word: str) -> str:
    if word[-1] == 'י':
        return word[:-1]
    return word

def make_groupings_linguistically(morph: List[str]) -> Tuple[List[List[int]], List[str]]:
    single_style_endings = {
        "אני": "י",
        "אתה": "ך",
        "את": "ך",
        "הוא": "ו",
        "היא": "ה",
        "אנחנו": "נו",
        "אתם": "כם",
        "אתן": "כן",
        "הם": "הם",
        "הן": "הן",
        }
    plural_style_endings = {
        "אני": "י",
        "אתה": "יך",
        "את": "יך",
        "הוא": "יו",
        "היא": "יה",
        "אנחנו": "ינו",
        "אתם": "יכם",
        "אתן": "יכן",
        "הם": "יהם",
        "הן": "יהן",
        }
    pronouns = single_style_endings.keys()
    
    sentence: List[str] = [morph[0]]
    groups = [[0]]
    
    SKIP_WORD = '**SKIP**'
    
    for i in range(1, len(morph)):
        m_w = morph[i]
        if m_w == SKIP_WORD:
            continue
        if m_w in string.punctuation:
            sentence.append(morph[i])
            groups.append([i])
            continue
        prev_word: str = sentence[-1]
        if m_w == 'ה' and prev_word in 'בלכ':
            sentence[-1] = sentence[-1] + ''
            groups[-1].append(i)
        elif m_w == 'ה' and prev_word in 'משו':
            assert i < len(morph) - 1
            sentence[-1] = sentence[-1] + 'ה' + morph[i+1]
            groups[-1].extend([i, i+1])
            morph[i+1] = SKIP_WORD
        elif m_w in ['ל', 'ב', 'כ'] and prev_word in 'וש':
            assert i < len(morph) - 1
            conc = morph[i+1]
            inds_to_add = [i+1]
            if conc == 'ה':
                conc = (morph[i+2])
                morph[i+2] = SKIP_WORD
                inds_to_add.append(i+2)
                
            sentence[-1] = sentence[-1] + m_w + conc
            groups[-1].extend(inds_to_add)
            morph[i+1] = SKIP_WORD
        elif m_w == 'הכל' and prev_word in 'בלכ':
            sentence[-1] = sentence[-1] + 'כל'
            groups[-1].append(i)
        elif ((len(prev_word) == 1 and prev_word in 'בלכהשומ') or prev_word == 'כש'):
            sentence[-1] = sentence[-1] + morph[i]
            groups[-1].append(i)
        elif correct_final_letters(m_w) in pronouns:
            m_w = correct_final_letters(m_w)
            if prev_word in ['אצל', 'בגלל', 'בשביל', 'בעד', 'בתוך', 'זולת', 'ליד', 'כמות', 'של', 'מאת',
                            'למען', 'לעמת', 'לקראת', 'לשם', 'מול', 'נגד', 'נכח', 'ב', 'ל', 'לעבר']:
                prev_word = normalise_final_letters(prev_word)
                sentence[-1] = sentence[-1] + single_style_endings[m_w]
                groups[-1].append(i)
            elif prev_word == 'יד' and sentence[-2] == 'על':
                sentence[-1] = sentence[-1] + single_style_endings[m_w]
                groups[-1].append(i)
            elif (nrw := normalise_final_letters(remove_trailing_yud(prev_word))) in ['כלפ', 'ביד', 'בלעד', 'לגב', 'לפנ', 'בעקבות',
                                                                                        'על','עד','תחת','אחר', 'אל']:
                sentence[-1] = nrw
                sentence[-1] = sentence[-1] + plural_style_endings[m_w]
                groups[-1].append(i)
            elif prev_word == 'ממן' or prev_word == 'מ':
                prev_word = 'מ'
                from_endings = {
                "אני": "מני",
                "אתה": "מך",
                "את": "מך",
                "הוא": "מנו",
                "היא": "מנה",
                "אנחנו": "מנו",
                "אתם": "כם",
                "אתן": "כן",
                "הם": "הם",
                "הן": "הן",
                }
                sentence[-1] = sentence[-1] + from_endings[m_w]
                groups[-1].append(i)
            elif correct_final_letters(prev_word) == 'עם':
                sentence[-1] = 'את'
                ending = single_style_endings[m_w]
                if len(ending) == 2 and ending[0] == 'ה':
                    ending = ending[1]
                sentence[-1] = sentence[-1] + ending
                groups[-1].append(i)
            elif prev_word == 'את':
                sentence[-1] = 'אות'
                ending = single_style_endings[m_w]
                if len(ending) == 2 and ending[0] == 'ה':
                    ending = ending[1]
                sentence[-1] = sentence[-1] + ending
                groups[-1].append(i)
            elif prev_word == 'אות':
                ending = single_style_endings[m_w]
                if len(ending) == 2 and ending[0] == 'ה':
                    ending = ending[1]
                sentence[-1] = sentence[-1] + ending
                groups[-1].append(i)
            elif prev_word == 'כמו':
                if m_w == 'אני':
                    sentence[-1] = sentence[-1] + 'ני'
                else:
                    sentence[-1] = sentence[-1] + single_style_endings[m_w]
                groups[-1].append(i)
            elif prev_word == 'לפי':
                sentence[-1] = 'לפ' + plural_style_endings[m_w]
                groups[-1].append(i)
            elif prev_word in 'וש':
                sentence[-1] = sentence[-1] + morph[i]
                groups[-1].append(i)
            else:
                sentence.append(morph[i])
                groups[-1].append(i)
        else:
            groups.append([i])
            sentence.append(m_w)
    return groups, sentence

def make_spans(labels: Iterable[str]) -> List[str]:
    """Returns label spans from labels for the purposes of evaluation. 

    Args:
        labels (list[str]): list of labels

    Returns:
        list str: format is category@low,high
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


def align_morph_to_tok(morph_labels: List[str], morphemes: List[str], sentence: List[str], multi_delim='^', validate_to_single=True) -> List[str]:
    # this one maybe for inputs?
    _, md = yap.yap_joint_api('\n'.join(sentence))
    md = yap.aggregate_morph(md)
    lings, words = make_groupings_linguistically(morphemes)
    # print(words)
    m_lings = max(map(max, lings)) + 1
    labels = []
    m_yap = md['FROM'].apply(lambda x: max(x)).max() + 1
    num_labs = len(morph_labels) 
    pad_size = 0
    if m_yap > num_labs: # more forms than labels
        padding_list = ['O'] * (pad_size := (m_yap - num_labs))
        morph_labels = padding_list + morph_labels        
    for i, (gy, gl) in enumerate(zip(md['FROM'], lings)):
        # print(f"gy: {gy}, gl: {gl}")
        label = multi_delim.join(morph_labels[i] for i in gy)
        label_l = multi_delim.join(morph_labels[i] for i in gl)
        if label != label_l:
            if words[i] == sentence[i]:
                # print('choosing this', label_l, words[i], sentence[i])
                label = label_l
        if validate_to_single:
            label, _ = validate_multi_to_single(label, multi_delim)
        labels.append(label)
    return labels


def evaluate_token_ner(pred: List[str], gold: List[str], multi_tok=False, multi_delim='^', beta=1):
    '''
    (Tjong Kim Sang and De Meulder 2003) for evaluation.
    
    Precision is the percentage of named entities found by the learning system that are correct.
    
    Recall is the percentage of named entities present in the corpus that are found by the system.
    
    A named entity is correct only if it is an exact match of the corresponding entity in the data file.
    
    Paper defaults beta to 1 for f_beta score
    
    `multi_tok` validates multi labels to single
    '''
    
    # we can ignore BIOSE labels, only interested in category and location
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
    
    return EvaluationMetrics(
        precision, recall, f_beta
    )
    
    
def evaluate_token_ner_nested(pred: List[List[str]], gold: List[List[str]], multi_tok=False, multi_delim='^', beta=1):
    assert len(pred) == len(gold)
    print("Evaluating performance...")
    
    corr_toks = 0
    tot = 0
    for pp, gg in zip(pred, gold):
        tot += len(pp)
        corr_toks += sum(p == g for p,g in zip(pp, gg))
    
    print(f"Correct tokens: {corr_toks}, Total tokens: {tot}, Accuracy: {corr_toks / tot:.4f}")
    
    correct = 0
    pred_len = 0
    gold_len = 0
    
    for p, g in zip(pred, gold):
        if not multi_tok:
            pred_span, gold_span = make_spans(p), make_spans(g)
        else:
            lam: Callable[[str], str] = lambda x: validate_multi_to_single(x, multi_delim)[0]
            pred_span, gold_span = make_spans(map(lam, p)), make_spans(map(lam, g))
        # Precision is the percentage of named entities found by the learning system that are correct.
        # Recall is the percentage of named entities present in the corpus that are found by the system.
        correct += len(set(pred_span).intersection(set(gold_span)))
        pred_len += len(pred_span)
        gold_len += len(gold_span)
        
    precision = correct / pred_len
    recall = correct / gold_len
    f_beta = ((beta**2 + 1) * precision * recall) / (beta**2 * precision + recall)
    
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F{beta}: {f_beta:.4f}")
    
    return EvaluationMetrics(
        precision, recall, f_beta
    )
    
    
def evaluate_morpheme(pred_morph: pd.DataFrame, morph: pd.DataFrame, multi: pd.DataFrame, tok: pd.DataFrame, multi_label_delim = '^'):
    '''
    Evaluates morph to morph and morph to single
    '''
    print("Morph to morph")
    m_to_m = evaluate_token_ner(pred_morph['Label'].to_list(), morph['Label'].to_list())
    
    merged = merge_morph_from_multi_spliting(pred_morph, multi, validate_to_single=True, multi_label_delim=multi_label_delim)
    
    print("\nMorph to single")
    m_to_multi = evaluate_token_ner(merged['Label'].to_list(), tok['Label'].to_list())
    
    return m_to_m, m_to_multi

def raw_toks_str_from_ner_df(df: pd.DataFrame) -> str:
    return '\n\n'.join(df
            .groupby('SentNum')['Word']
            .agg('\n'.join)) + '\n\n'
    
if __name__ == '__main__':
    PRED_MORPH = '/Users/yuval/GitHub/hebrew-ner/hpc_eval_results/morph_cnn_seed_50.txt'
    
    YAP_MORPH = '/Users/yuval/GitHub/hebrew-ner/hpc_eval_results/morph_cnn_seed_50_yap.txt'
    
    morph = read_file_to_sentences_df(MORPH)
    pred_morph = read_file_to_sentences_df(PRED_MORPH)
    multi = read_file_to_sentences_df(MULTI)
    tok = read_file_to_sentences_df(TOK)
    
    yap_morph = read_file_to_sentences_df(YAP_MORPH)
    
    # print(merge_morph_from_multi_spliting(yap_morph, multi))
    
    ORIGINS = '/Users/yuval/GitHub/hebrew-ner/utils_eval_files/yap_morph_dev_tokens.txt'
    
    origins = read_token_origins_to_df(ORIGINS)
    
    
    # for x,y in splitting_sents:
    #     for z in y:
    #         print(z)
        # print(y)
    
    print(merge_morph_from_token_origins(yap_morph, origins, validate_to_single=True))
  
    # print(read_file_to_sentences_df(YAP_MORPH))
    