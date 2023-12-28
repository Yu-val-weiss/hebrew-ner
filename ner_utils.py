import re
import string
from typing import Callable, Iterable, List, NamedTuple, Tuple
from utils.metric import get_ner_BMES, get_ner_fmeasure, fmeasure_from_file
import time

#TODO: write an accumulator to convert morpheme to token, based on gold unsegmented data

class WordLabel(NamedTuple):
    word: str
    label: str
    def concat(self, wl: 'WordLabel', label_delim = '^'):
        return WordLabel(self.word+wl.word, self.label + label_delim + wl.label)
    
class EvaluationMetrics(NamedTuple):
    precision: float
    recall: float
    f: float

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

# actual words don't matter, just lengths of each sentence should match so עמי vs אתי doesn't matter
def align_morph_to_tok(morph: List[WordLabel], tok: List[WordLabel]) -> List[WordLabel]:
    '''
    Linguistic info from https://hebrew-academy.org.il/2014/03/05/נטיית-מילות-היחס/
    '''
    # print(morph[0] == tok[0])
    i = 0
    new_word_labels = []
    for j, t in enumerate(tok):
        print(f"i: {i}, j: {j}")
        if morph[i].word == t.word:
            new_word_labels.append(morph[i])
            i += 1
        else:
            word, label = [morph[i].word], [morph[i].label]
            i += 1
            while (w := ''.join(word)) != t.word.replace("״", '"').replace("”", '"'):
                try:
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
                    m_w = morph[i].word
                    if m_w == 'ה' and word[-1] in 'בלכ':
                        word.append('')
                    elif m_w == 'הכל' and word[-1] in 'בלכ':
                        word.append('כל')  
                    elif correct_final_letters(m_w) in pronouns:
                        m_w = correct_final_letters(m_w)
                        if word[-1] in ['אצל', 'בגלל', 'בשביל', 'בעד', 'בתוך', 'זולת', 'ליד', 'כמות', 'של', 'מאת',
                                        'למען', 'לעמת', 'לקראת', 'לשם', 'מול', 'נגד', 'נכח', 'ב', 'ל', 'לעבר']:
                            word[-1] = normalise_final_letters(word[-1])
                            word.append(single_style_endings[m_w])
                        elif word[-1] == 'יד' and word[-2] == 'על':
                            word.append(single_style_endings[m_w])
                        elif (nrw := normalise_final_letters(remove_trailing_yud(word[-1]))) in ['כלפ', 'ביד', 'בלעד', 'לגב', 'לפנ', 'בעקבות',
                                                                                                 'על','עד','תחת','אחר', 'אל']:
                            word[-1] = nrw
                            word.append(plural_style_endings[m_w])
                        elif word[-1] == 'ממן' or word[-1] == 'מ':
                            word[-1] = 'מ'
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
                            word.append(from_endings[m_w])
                        elif correct_final_letters(word[-1]) == 'עם':
                            word[-1] = 'את'
                            ending = single_style_endings[m_w]
                            if len(ending) == 2 and ending[0] == 'ה':
                                ending = ending[1]
                            word.append(ending)
                        elif word[-1] == 'את':
                            word[-1] = 'אות'
                            ending = single_style_endings[m_w]
                            if len(ending) == 2 and ending[0] == 'ה':
                                ending = ending[1]
                            word.append(ending)
                        elif word[-1] == 'אות':
                            word[-1] = ''
                            ending = single_style_endings[m_w]
                            if len(ending) == 2 and ending[0] == 'ה':
                                ending = ending[1]
                            word.append(ending)
                        elif word[-1] == 'כמו':
                            if m_w == 'אני':
                                word.append('ני')
                            else:
                                word.append(single_style_endings[m_w])
                        elif word[-1] == 'לפי':
                            word[-1] = 'לפ'
                            word.append(plural_style_endings[m_w])
                        else:
                            word.append(m_w)
                                
                    else:
                         word.append(m_w)
                    label.append(morph[i].label)
                    i += 1
                except:
                    print(f"Oops {i}: {w[:10]}, {j}: {t.word}")
                    raise

            new_word_labels.append(WordLabel(w, '^'.join(label)))
    
        print(f"Just appended: {new_word_labels[-1]}")
    
    return new_word_labels



def align_morph_sentence_to_tok(morph: List[WordLabel]) -> List[WordLabel]:
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
    
    sentence = [morph[0]]
    
    SKIP_WORD = '**SKIP**'
    
    for i in range(1, len(morph)):
        m_w, m_l = morph[i]
        if m_w == SKIP_WORD:
            continue
        if m_w in string.punctuation:
            sentence.append(morph[i])
            continue
        prev_word = sentence[-1].word
        if m_w == 'ה' and prev_word in 'בלכ':
            sentence[-1] = sentence[-1].concat(WordLabel('', m_l))
        elif m_w == 'ה' and prev_word in 'משו':
            assert i < len(morph) - 1
            sentence[-1] = sentence[-1].concat(WordLabel('ה', m_l)).concat(morph[i+1])
            morph[i+1] = WordLabel(SKIP_WORD, '')
        elif m_w in ['ל', 'ב', 'כ'] and prev_word in 'וש':
            assert i < len(morph) - 1
            conc = morph[i+1]
            if conc.word == 'ה':
                conc = WordLabel('', conc.label)
                conc = conc.concat(morph[i+2])
                morph[i+2] = WordLabel(SKIP_WORD, '')
            sentence[-1] = sentence[-1].concat(WordLabel(m_w, m_l)).concat(conc)
            morph[i+1] = WordLabel(SKIP_WORD, '')
        elif m_w == 'הכל' and prev_word in 'בלכ':
            sentence[-1] = sentence[-1].concat(WordLabel('כל', m_l))
        elif ((len(prev_word) == 1 and prev_word in 'בלכהשומ') or prev_word == 'כש'):
            sentence[-1] = sentence[-1].concat(morph[i])
        elif correct_final_letters(m_w) in pronouns:
            m_w = correct_final_letters(m_w)
            if prev_word in ['אצל', 'בגלל', 'בשביל', 'בעד', 'בתוך', 'זולת', 'ליד', 'כמות', 'של', 'מאת',
                            'למען', 'לעמת', 'לקראת', 'לשם', 'מול', 'נגד', 'נכח', 'ב', 'ל', 'לעבר']:
                prev_word = normalise_final_letters(prev_word)
                sentence[-1] = sentence[-1].concat(WordLabel(single_style_endings[m_w], m_l))
            elif prev_word == 'יד' and sentence[-2].word == 'על':
                sentence[-1] = sentence[-1].concat(WordLabel(single_style_endings[m_w], m_l))
            elif (nrw := normalise_final_letters(remove_trailing_yud(prev_word))) in ['כלפ', 'ביד', 'בלעד', 'לגב', 'לפנ', 'בעקבות',
                                                                                        'על','עד','תחת','אחר', 'אל']:
                sentence[-1] = WordLabel(nrw, sentence[-1].label)
                sentence[-1] = sentence[-1].concat(WordLabel(plural_style_endings[m_w], m_l))
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
                sentence[-1] = sentence[-1].concat(WordLabel(from_endings[m_w], m_l))
            elif correct_final_letters(prev_word) == 'עם':
                sentence[-1] = WordLabel('את', sentence[-1].label)
                ending = single_style_endings[m_w]
                if len(ending) == 2 and ending[0] == 'ה':
                    ending = ending[1]
                sentence[-1] = sentence[-1].concat(WordLabel(ending, m_l))
            elif prev_word == 'את':
                sentence[-1] = WordLabel('אות', sentence[-1].label)
                ending = single_style_endings[m_w]
                if len(ending) == 2 and ending[0] == 'ה':
                    ending = ending[1]
                sentence[-1] = sentence[-1].concat(WordLabel(ending, m_l))
            elif prev_word == 'אות':
                ending = single_style_endings[m_w]
                if len(ending) == 2 and ending[0] == 'ה':
                    ending = ending[1]
                sentence[-1] = sentence[-1].concat(WordLabel(ending, m_l))
            elif prev_word == 'כמו':
                if m_w == 'אני':
                    sentence[-1] = sentence[-1].concat(WordLabel('ני', m_l))
                else:
                    sentence[-1] = sentence[-1].concat(WordLabel(single_style_endings[m_w], m_l))
            elif prev_word == 'לפי':
                prev_label = sentence[-1].label
                sentence[-1] = WordLabel(
                    'לפ' + plural_style_endings[m_w],
                    prev_label
                )
            elif prev_word in 'וש':
                sentence[-1] = sentence[-1].concat(morph[i])
            else:
                sentence.append(morph[i])
                # sentence[-1] = sentence[-1].concat(WordLabel(m_w, m_l))
         
       
         
        # elif prev_word == 'ה' and len(m_w) > 8:
        #     sentence[-1] = sentence[-1].concat(WordLabel(m_w, m_l))
                   
        else:
            sentence.append(WordLabel(m_w, m_l))
            
            
    return sentence

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
    
    return EvaluationMetrics(
        precision, recall, f_beta
    )
    
    
    
if __name__ == '__main__':
    # res = read_file("hpc_eval_results/morph_cnn_seed_50.txt")
    # pred_labels = [x.label for x in res]
    # gold = read_file("/Users/yuval/GitHub/NEMO-Corpus/data/spmrl/gold/morph_gold_dev.bmes")
    
    morph = read_file_to_sentences("/Users/yuval/GitHub/NEMO-Corpus/data/spmrl/gold/morph_gold_test.bmes")
    multi_tok = read_file_to_sentences("/Users/yuval/GitHub/NEMO-Corpus/data/spmrl/gold/token-multi_gold_test.bmes")
    tok = read_file("/Users/yuval/GitHub/NEMO-Corpus/data/spmrl/gold/token-single_gold_dev.bmes")
    
    start = time.time()
    
    # new_morph = align_morph_to_tok(morph, tok)
    
    new_morph = map(align_morph_sentence_to_tok, morph)
    
    end = time.time()
    
    print(f"Align time: {end-start}")
    
    count = 0
    for mrp, mlt in zip(new_morph, multi_tok):
        if abs(len(mrp) - len(mlt)) > 1:
            print("Got one wrong :()")
            print(mrp)
            print(mlt)
            print("Diff in size", len(mrp) - len(mlt))
            count += 1
        
    
    print("Accuracy: ", (len(multi_tok) - count) / len(multi_tok))
    
    # for n_m, m in zip(new_morph, multi_tok):
    #     print("Test", n_m, m)
    
    # gold = [x.label for x in gold]
    
    
    
    # evaluate_token_ner(pred_labels, gold, multi_tok=True)