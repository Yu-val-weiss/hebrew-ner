from enum import Enum
import os
import tempfile
import time
from typing import Dict, Generator, List, Set, Tuple
import torch
from model.seqlabel import SeqLabel
from ncrf_main import evaluate
from utils import ner, yap
from utils.data import Data
from utils.yap_graph import YapGraph
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel
from utils.tokenizer import text2listOfSentences, tokenize_sentences

class ModelEnum(str, Enum):
    token_single = 'token_single'
    token_multi = 'token_multi'
    morph = 'morph'
    hybrid = 'hybrid'
    morph_lstm_ftam = 'morph_lstm_ftam'


models: Dict[str, Tuple[Data, SeqLabel]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load ML models
    for model_name in ModelEnum:
        path = os.path.join("api_configs", f"{model_name}.conf")
        if not os.path.exists(path):
            continue
        print(f"Creating model {model_name}...")
        data = Data()
        data.HP_gpu = torch.cuda.is_available()
        data.read_config(f'api_configs/{model_name}.conf')
        data.load(data.dset_dir, fasttext_model_dir='fasttext/wiki.he.bin')
        data.read_config(f'api_configs/{model_name}.conf') # need to reload for model dir and stuff like that
        if not torch.cuda.is_available():
            data.HP_gpu = False
        model = SeqLabel(data)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(data.load_model_dir)) # type: ignore
        else:
            model.load_state_dict(torch.load(data.load_model_dir, map_location=torch.device('cpu'))) # type: ignore

        models[model_name] = (data, model)
    
    yield
    models.clear()
    

app = FastAPI(debug=True, lifespan=lifespan)

  
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
    

class NERQuery(BaseModel):
    text: str
    model: ModelEnum = ModelEnum.token_single
    

class NERLabelledToken(BaseModel):
    token: str
    label: str
    
class NERResponse(BaseModel):
    prediction: List[List[NERLabelledToken]]
    
class TokenizeQuery(BaseModel):
    text: str

class NamedTemporary:
    def __enter__(self) -> str:
        self.file = tempfile.NamedTemporaryFile(prefix='heb-ner-tmp-', encoding='utf-8', mode="r")
        return self.file.name
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()


@app.get('/')
def home():
    return "OK"

@app.post("/tokenize")
def api_tokenize(q: TokenizeQuery):
    sents = text2listOfSentences(q.text)
    return tokenize_sentences(sents)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


def predict_text(data: Data, model: SeqLabel, text: List[List[str]]) -> List[List[str]]:
    with NamedTemporary() as tmpfile:
        with open(tmpfile, 'w', encoding='utf-8') as tmpf:
            for sent in text:
                tmpf.write('\n'.join(map(lambda x: x + '\tO',sent)))
                tmpf.write('\n\n')
        
        data.raw_dir = tmpfile
        data.generate_instance('raw')
        speed, pred_results, pred_scores = evaluate(data, model, 'raw', skip_eval=True)  # type: ignore
        
    return pred_results # type: ignore

def predict_text_from_md_df(data: Data, model: SeqLabel, md: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns ner dataframe
    '''
    text = md.copy(deep=True)
    
    text.rename(
        columns = {
        'FORM': 'Token',
        'SENTNUM': 'SentNum'
    }, inplace=True)
    
    text['WordIndex'] = text.groupby('SentNum').cumcount()
    
    with NamedTemporary() as tmpfile:
        with open(tmpfile, 'w', encoding='utf-8') as tmpf:
            tmpf.write('\n\n'.join(text.groupby('SentNum')['Token'].agg(lambda x: '\n'.join(map(lambda x: x + ' O', x)))))
        data.raw_dir = tmpfile
        data.generate_instance('raw')
        speed, pred_results, pred_scores = evaluate(data, model, 'raw', skip_eval=True)  # type: ignore
        
        
    def pred_result_gen(pred_result):
        for sent_num, sent in enumerate(pred_result):
            for word_index, label in enumerate(sent):
                yield (sent_num, word_index, label)
        
    label_df = pd.DataFrame(
        data = pred_result_gen(pred_results),
        columns = ['SentNum', 'WordIndex', 'Label']
    )
    
    return pd.merge(text, label_df, on=['SentNum', 'WordIndex'])[['SentNum', 'WordIndex', 'Token', 'Label']]


def standard_predict(data: Data, model: SeqLabel, text: List[List[str]]) -> NERResponse:
    pred_results = predict_text(data, model, text)
    
    # print(pred_results)
    # print(type(pred_results))
    # print(type(pred_results[0]))
        
    return wrap_pred_text_response(text, pred_results)


def wrap_pred_text_response(text, pred_results):
    prediction_result = []
    for row1, row2 in zip(text, pred_results): 
        tokens_labels = [NERLabelledToken(token=t, label=l) for t, l in zip(row1, row2)]
        prediction_result.append(tokens_labels)

    return NERResponse(
        prediction=prediction_result
    )
    
    
def pred_result_to_df(text: List[List[str]], pred_result: List[List[str]]) -> pd.DataFrame:
    def df_gen() -> Generator[Tuple[int, int, str, str], None, None]:
        for sent_num, (text_sent, pred_sent) in enumerate(zip(text, pred_result)):
            for word_ind, (t, p) in enumerate(zip(text_sent, pred_sent)):
                yield (sent_num, word_ind, t, p)

    return pd.DataFrame(
        data = df_gen(),
        columns = ner.NER_DF_COLUMNS
    )
    
def hybrid_predict(text: List[List[str]]) -> NERResponse:
    if ModelEnum.token_multi not in models or ModelEnum.morph not in models:
        raise HTTPException(status_code=404, detail="The requisite models were not properly loaded by the API.")
    
    tokenized_text_as_str = '\n\n'.join(map('\n'.join, text)) + '\n\n'
    
    multi_data, multi_model = models[ModelEnum.token_multi]
    
    multi_pred_result = predict_text(multi_data, multi_model, text)
    
    multi = pred_result_to_df(text, multi_pred_result)
    
    ma = yap.yap_ma_api(tokenized_text_as_str)
    
    print("MA complete")
    
    pruned = prune_lattices(ma, multi, fallback=True)
    
    md = yap.yap_joint_from_lattice_api(pruned)
    
    morph_data, morph_model = models[ModelEnum.morph]
    
    morph_from_pruned = predict_text_from_md_df(morph_data, morph_model, md)
    
    tok_origins = yap.md_to_origins_df(md)
    
    res = ner.merge_morph_from_token_origins(morph_from_pruned, tok_origins, validate_to_single=True)
    
    # fix text at the end
    
    res['Token'] = multi['Word']
    
    res['LabelToken'] = res.apply(lambda x: NERLabelledToken(token=x['Token'], label=x['Label']), axis=1) # type: ignore

    return NERResponse(
        prediction = res.groupby('SentNum')['LabelToken'].agg(list).to_list()
    )



@app.post('/predict')
def api_predict(q: NERQuery) -> NERResponse:
    tt = tokenize_sentences(text2listOfSentences(q.text))
    if q.model == ModelEnum.hybrid:
        return hybrid_predict(tt)
    
    if q.model not in models:
        raise HTTPException(status_code=404, detail=f"The model '{q.model}' has not been loaded, please try one of {list(models.keys())}")
    data, model = models[q.model]
    
    return standard_predict(data, model, tt)
    


@app.get('/healthcheck')
def health():
    return "OK"

if __name__ == '__main__':
    uvicorn.run('ner_app:app', port=5000, reload=True, reload_excludes=["ner_app_tests.py"])