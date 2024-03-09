# -*- coding: utf-8 -*-
# @Author: me
from typing import List, Set
import requests
import pandas as pd
from types import SimpleNamespace
from app_env import ENV

YAP_PATH = f"http://{ENV.YAP_HOST}:{ENV.YAP_PORT}/yap/heb"

LATTICE_COLUMNS = ['SENTNUM', 'FROM', 'TO', 'FORM', 'LEMMA', 'C_POS_TAG', 'POS_TAG', 'FEATS', 'TOKEN']
'''
SENTNUM: Index of the sentence
FROM: Index of the outgoing vertex of the edge
TO: Index of the incoming vertex of the edge
FORM: word form or punctuation mark
LEMMA: Lemma of the word form; underscore if not available
C_POS_TAG: Coarse-grained part-of-speech tag; underscore if not available
POS_TAG: Fine-grained part-of-speech tag; underscore if not available; in YAP both POSTAG and CPOSTAG are always identical
FEATS: List of morphological features separated by a vertical bar (|) from a pre-defined language-specific inventory; underscore if not available
TOKEN: Source token index
'''
CONLL_COLUMNS = ['SENTNUM', 'ID', 'FORM', 'LEMMA', 'C_POS_TAG', 'POS_TAG', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']
'''
SENTNUM: Index of the sentence
ID: Morpheme index, starting at 1 for each new sentence
FORM: Word form or punctuation mark
LEMMA: Lemma of word form; underscore if not available
C_POS_TAG: Coarse-grained part-of-speech tag; underscore if not available
POS_TAG: Fine-grained part-of-speech tag; underscore if not available; in YAP both POSTAG and CPOSTAG are always identical
FEATS: List of morphological features separated by a vertical bar (|) from a pre-defined language-specific inventory; underscore if not available
HEAD: Head of the current morpheme, which is either a value of ID, or zero (’0’) if the token links to the virtual root node of the sentence. There may be multiple tokens with a HEAD value of zero.
DEPREL: Dependency relation to the HEAD. The dependency relation of a token with HEAD=0 is simply ’ROOT’
PHEAD: Projective head; Not relevant - YAP doesn't use it
PDEPREL: Dependency relation to the PHEAD; not relevant - YAP doesn't use it
'''
    
def yap_joint_api(text: str):
    '''
    Returns `(ma, md)` Dataframes 
    
    Columns are `['SENTNUM', 'FROM', 'TO', 'FORM', 'LEMMA', 'C_POS_TAG', 'POS_TAG', 'FEATS', 'TOKEN']`
        SENTNUM: Index of the sentence
        FROM: Index of the outgoing vertex of the edge
        TO: Index of the incoming vertex of the edge
        FORM: word form or punctuation mark
        LEMMA: Lemma of the word form; underscore if not available
        C_POS_TAG: Coarse-grained part-of-speech tag; underscore if not available
        POS_TAG: Fine-grained part-of-speech tag; underscore if not available; in YAP both POSTAG and CPOSTAG are always identical
        FEATS: List of morphological features separated by a vertical bar (|); underscore if not available
        TOKEN: Source token index
    '''
    text = text.strip() + '  ' # need 2 spaces at end
    # NOTE: for multiple sentences, need double space or \n after full stop.
    
    payload = {"text": text}
    headers = {"content-type": "application/json"}

    r = requests.get(f"{YAP_PATH}/joint", json=payload, headers=headers)

    r.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    
    response_data = r.json(object_hook=lambda x: SimpleNamespace(**x))
    
    # dt_df = make_data_frame_from_yap_str(response_data.dep_tree, columns=CONLL_COLUMNS)
    
    ma_df = make_data_frame_from_yap_str(response_data.ma_lattice)
    
    md_df = make_data_frame_from_yap_str(response_data.md_lattice)
    
    return ma_df, md_df


def yap_joint_from_lattice_api(lattice: pd.DataFrame):
    '''
    Input `ma` lattice Dataframe
    
    Returns `(md)` Dataframe
    
    Columns are `['SENTNUM', 'FROM', 'TO', 'FORM', 'LEMMA', 'C_POS_TAG', 'POS_TAG', 'FEATS', 'TOKEN']`
        SENTNUM: Index of the sentence
        FROM: Index of the outgoing vertex of the edge
        TO: Index of the incoming vertex of the edge
        FORM: word form or punctuation mark
        LEMMA: Lemma of the word form; underscore if not available
        C_POS_TAG: Coarse-grained part-of-speech tag; underscore if not available
        POS_TAG: Fine-grained part-of-speech tag; underscore if not available; in YAP both POSTAG and CPOSTAG are always identical
        FEATS: List of morphological features separated by a vertical bar (|); underscore if not available
        TOKEN: Source token index
    '''
    s = lattice_df_to_yap_str(lattice)
    payload = {"amb_lattice": s}
    headers = {"content-type": "application/json"}

    r = requests.get(f"{YAP_PATH}/joint/lattice", json=payload, headers=headers)

    r.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    
    response_data = r.json(object_hook=lambda x: SimpleNamespace(**x))
    
    md_df = make_data_frame_from_yap_str(response_data.md_lattice)
    
    return md_df
    

def yap_ma_api(text: str):
    '''
    Returns `ma` Dataframe
    
    Columns are `['SENTNUM', 'FROM', 'TO', 'FORM', 'LEMMA', 'C_POS_TAG', 'POS_TAG', 'FEATS', 'TOKEN']`
        SENTNUM: Index of the sentence
        FROM: Index of the outgoing vertex of the edge
        TO: Index of the incoming vertex of the edge
        FORM: word form or punctuation mark
        LEMMA: Lemma of the word form; underscore if not available
        C_POS_TAG: Coarse-grained part-of-speech tag; underscore if not available
        POS_TAG: Fine-grained part-of-speech tag; underscore if not available; in YAP both POSTAG and CPOSTAG are always identical
        FEATS: List of morphological features separated by a vertical bar (|); underscore if not available
        TOKEN: Source token index
    '''
    text = text.strip() + '  ' # need 2 spaces at end
    # NOTE: for multiple sentences, need double space or \n after full stop.
    
    payload = {"text": text}
    headers = {"content-type": "application/json"}

    r = requests.get(f"{YAP_PATH}/ma", json=payload, headers=headers)

    r.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    
    response_data = r.json(object_hook=lambda x: SimpleNamespace(**x))
    
    ma_df = make_data_frame_from_yap_str(response_data.ma_lattice)
    
    return ma_df


def make_data_frame_from_yap_str(df_str: str, columns: List[str]=LATTICE_COLUMNS, numeric_cols: Set[str] = {'ID', 'FROM', 'TO', 'HEAD', 'TOKEN'}):
    '''
    Columns are `['SENTNUM', 'FROM', 'TO', 'FORM', 'LEMMA', 'C_POS_TAG', 'POS_TAG', 'FEATS', 'TOKEN']`
        SENTNUM: Index of the sentence
        FROM: Index of the outgoing vertex of the edge
        TO: Index of the incoming vertex of the edge
        FORM: word form or punctuation mark
        LEMMA: Lemma of the word form; underscore if not available
        C_POS_TAG: Coarse-grained part-of-speech tag; underscore if not available
        POS_TAG: Fine-grained part-of-speech tag; underscore if not available; in YAP both POSTAG and CPOSTAG are always identical
        FEATS: List of morphological features separated by a vertical bar (|); underscore if not available
        TOKEN: Source token index
    '''
    dfs = []
    for sent_num, df_s in enumerate(df_str.strip().split('\n\n')):
        dt_df = pd.DataFrame(
            columns=columns, 
            data=(([sent_num] + l.split('\t')) for l in df_s.strip().split("\n"))
        )
        # dt_df['FEATS'] = dt_df['FEATS'].apply(lambda x: [] if x == '_' else x.split('|'))
        dt_df = dt_df.apply(lambda col: pd.to_numeric(col) if col.name in numeric_cols else col)
        dfs.append(dt_df)

    return pd.concat(dfs).reset_index(drop=True)


def aggregate_morph(morph_disamb_df: pd.DataFrame):
    return (morph_disamb_df
            .groupby('TOKEN', sort=False)
            .agg(list)
            .reset_index()
            )
    

def md_to_origins_df(md: pd.DataFrame):
    '''
    Input `md` lattice Dataframe
    
    Returns `(origins)` Dataframe
    
    Reads a file of sentences, with each word NER labelled into a pandas dataframe
    
    Note: Yap assigns 1-based indices, so will subtract 1 here 
    
    Columns: `['SentNum', 'WordIndex', 'Origin']` 
    
    Can use `r.groupby('SentNum').agg(list)` to get them aggregated into sentences
    '''
    origins = md.copy(deep=True)
    
    origins.rename(
        columns={
            'SENTNUM': 'SentNum',
            'TOKEN': 'Origin'
        },
        inplace=True
    )
    
    origins['WordIndex'] = origins.groupby('SentNum').cumcount()
    
    origins['Origin'] = origins['Origin'] - 1
    
    origins = origins[['SentNum', 'WordIndex', 'Origin']]
    
    return origins
    
    
def lattice_df_to_yap_str(lattice: pd.DataFrame):
    x = [df
         .drop('SENTNUM', axis='columns')
         .to_csv(header=False, index=False, sep='\t', quotechar="'")
         for _, df
         in lattice.groupby('SENTNUM')]
    return '\n\n'.join(x).strip() + '\n\n' 


if __name__ == '__main__':
    s = "עשרות אנשים מגעים מתאילנד לישראל כשהם נרשמים כמתנדבים , אך למעשה משמשים עובדים שכירים זולים .  "
    s2 = "עשרות אנשים מגעים מתאילנד לישראל כשהם נרשמים כמתנדבים , אך למעשה משמשים עובדים שכירים זולים .  גנו גידל דגן בגן .  "
    ganan = "גנן גידל דגן בגן .  "
    # ma, md =  yap_joint_api(s2)
    _, md = yap_joint_api(s2)
    
    
    
    print(md.head())

    # origins = md_to_origins_df(md)
    
    # print(origins.groupby('SentNum')['Origin'].agg(list).to_list())
    
    # md = md.groupby('SENTNUM')['FORM'].agg('\n'.join)
    # with open('test.txt', 'w') as w:
    #     w.write('\n\n'.join(md) + '\n\n')
    