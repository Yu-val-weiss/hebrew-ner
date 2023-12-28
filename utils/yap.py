# -*- coding: utf-8 -*-
# @Author: Yuval Weiss
from typing import List, Set
import requests
import pandas as pd
from types import SimpleNamespace

YAP_PATH = "http://localhost:8000/yap/heb"
LATTICE_COLUMNS = ['FROM', 'TO', 'FORM', 'LEMMA', 'C_POS_TAG', 'POS_TAG', 'FEATS', 'TOKEN']
'''
FROM: Index of the outgoing vertex of the edge
TO: Index of the incoming vertex of the edge
FORM: word form or punctuation mark
LEMMA: Lemma of the word form; underscore if not available
C_POS_TAG: Coarse-grained part-of-speech tag; underscore if not available
POS_TAG: Fine-grained part-of-speech tag; underscore if not available; in YAP both POSTAG and CPOSTAG are always identical
FEATS: List of morphological features separated by a vertical bar (|) from a pre-defined language-specific inventory; underscore if not available
TOKEN: Source token index
'''
CONLL_COLUMNS = ['ID', 'FORM', 'LEMMA', 'C_POS_TAG', 'POS_TAG', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']
'''
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
SEGMENTATION_COLUMNS = ['Split']
    
def yap_joint_api(text: str):
    text = text.strip() + '  ' # need 2 spaces at end
    
    payload = {"text": text}
    headers = {"content-type": "application/json"}

    r = requests.get(f"{YAP_PATH}/joint", json=payload, headers=headers)

    r.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    
    response_data = r.json(object_hook=lambda x: SimpleNamespace(**x))
    
    dt_df = make_data_frame_from_yap_str(response_data.dep_tree, columns=CONLL_COLUMNS)
    
    ma_df = make_data_frame_from_yap_str(response_data.ma_lattice)
    
    md_df = make_data_frame_from_yap_str(response_data.md_lattice)
    
    return dt_df, ma_df, md_df

def make_data_frame_from_yap_str(df_str: str, columns: List[str]=LATTICE_COLUMNS, numeric_cols: Set[str] = {'ID', 'FROM', 'TO', 'HEAD', 'TOKEN'}):
    dt_df = pd.DataFrame(
        columns=columns, 
        data=(l.split('\t') for l in df_str.split("\n")[:-2]) # remove the last two spaces
    )
    
    dt_df['FEATS'] = dt_df['FEATS'].apply(lambda x: [] if x == '_' else x.split('|'))
    dt_df = dt_df.apply(lambda col: pd.to_numeric(col) if col.name in numeric_cols else col)
    return dt_df

def aggregate_morph(morph_disamb_df: pd.DataFrame, columns: List[str]=['FROM', 'FORM', 'TOKEN']):
    r = (morph_disamb_df
            .groupby('TOKEN', sort=False)
            .agg(list)
            .reset_index()
            )[columns]
    # print(r)
    return r


if __name__ == '__main__':
    s = "עשרות אנשים מגעים מתאילנד לישראל כשהם נרשמים כמתנדבים , אך למעשה משמשים עובדים שכירים זולים .  "
    _, _, md =  yap_joint_api(s)
    r = aggregate_morph(md)