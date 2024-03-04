import time
from typing import List, Set, Tuple, Union
import torch
from model.seqlabel import SeqLabel
from ncrf_main import evaluate, load_model_decode
from utils import yap, ner, functions
from utils.data import Data
from utils.yap_graph import YapGraph
import pandas as pd
from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel
import fasttext


models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    data = Data()
    data.HP_gpu = torch.cuda.is_available()
    data.read_config('api_configs/trn_tok_sing.conf')
    data.load(data.dset_dir)
    data.read_config('api_configs/trn_tok_sing.conf') # need to reload for model dir and stuff like that
    if not torch.cuda.is_available():
        data.HP_gpu = False
    model = SeqLabel(data)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(data.load_model_dir)) # type: ignore
    else:
        model.load_state_dict(torch.load(data.load_model_dir, map_location=torch.device('cpu'))) # type: ignore

    models['token_single'] = (data, model)
    
    yield
    models.clear()

app = FastAPI(debug=True, lifespan=lifespan)


def read_instance(in_words, word_alphabet, char_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>', fasttext: Union[None, fasttext.FastText._FastText] = None):
    '''
    in_words: input words
    '''
    instance_texts = []
    instance_ids = []
    chars = []
    labels = []
    word_Ids = []
    char_Ids = []
    label_Ids = []
    
    for word in in_words:
        if number_normalized:
            word = functions.normalize_word(word)
        if fasttext is not None:
            word_Ids.append(fasttext.get_word_vector(word))
        else:
            word_Ids.append(word_alphabet.get_index(word))
        ## get char
        char_list = []
        char_Id = []
        for char in word:
            char_list.append(char)
        if char_padding_size > 0:
            char_number = len(char_list)
            if char_number < char_padding_size:
                char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
            assert(len(char_list) == char_padding_size)
        else:
            ### not padding
            pass
        for char in char_list:
            char_Id.append(char_alphabet.get_index(char))
        chars.append(char_list)
        char_Ids.append(char_Id)
    if (len(in_words) > 0) and ((max_sent_length < 0) or (len(in_words) < max_sent_length)) :
        instance_texts.append([in_words, [[]], chars, [['O']*len(in_words)]])
        instance_ids.append([word_Ids, [[]], char_Ids, [[0]*len(in_words)]])
        words = []
        chars = []
        labels = []
        word_Ids = []
        char_Ids = []
        label_Ids = []
    return instance_texts, instance_ids


def predict(data: Data, model: SeqLabel, in_words: List[str]):
    data.raw_texts, data.raw_Ids = read_instance(in_words, data.word_alphabet, data.char_alphabet, data.number_normalized, data.MAX_SENTENCE_LENGTH, fasttext=data.fasttext_model)
    name = 'raw'
    print("Decode %s data, nbest: %s ..."%(name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    if data.seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
    return pred_results, pred_scores
    
    
def prune_lattices(lattice_df: pd.DataFrame, multi_df: pd.DataFrame, multi_label_delim='^', fallback=False):
    splitting = ner.make_multi_splitting_df(multi_df, multi_label_delim)
    valid_edges: Set[Tuple[int, int, int, int]] = set()
    for ((sent_id, tok_id), sub_lattice),(_, split) in zip(lattice_df.groupby(['SENTNUM', 'TOKEN']), splitting.groupby(['SentNum', 'WordIndex'])):
        # tok_id in lattice starts from 1, so may need to offset
        g = YapGraph.from_df(sub_lattice)
        source, target = sub_lattice['FROM'].iat[0], sub_lattice['TO'].iat[-1]
        path_len = split['Splitting'].iat[0] + 1
        paths = list(g.get_all_paths(source, target, limit=path_len))
        pruned_paths = [p for p in paths if len(p) == path_len]
        if fallback and len(pruned_paths) == 0:
            pruned_paths = [p for p in paths if abs(len(p) - path_len) <= 1]
        if len(pruned_paths) > 0:
            paths = pruned_paths
        for p in paths:
            for f, t in zip(p[:-1], p[1:]):
                valid_edges.add((sent_id, tok_id, f, t))
                
    cols_to_filter = ['SENTNUM', 'TOKEN', 'FROM', 'TO']
    return lattice_df[lattice_df[cols_to_filter].apply(tuple, axis=1).isin(valid_edges)].reset_index(drop=True)
    

@app.get('/')
def home():
    return "OK"


@app.get('/test')
def test():
    data, model = models['token_single']
    return predict(data, model, ['גנו'])


@app.get('/healthcheck')
def health():
    return "OK"




if __name__ == '__main__':
    uvicorn.run('ner_app:app', port=5000, reload=True)