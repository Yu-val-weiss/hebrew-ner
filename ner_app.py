from enum import Enum
import os
import tempfile
import time
from typing import Dict, List, Set, Tuple
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
        data.read_config('api_configs/token_single.conf')
        data.load(data.dset_dir)
        data.read_config('api_configs/token_single.conf') # need to reload for model dir and stuff like that
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

@app.get('/test')
def test():
    with NamedTemporary() as tmpfile:
        return (tmpfile)

@app.post("/tokenize")
def api_tokenize(q: TokenizeQuery):
    sents = text2listOfSentences(q.text)
    return sents, tokenize_sentences(sents)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


def standard_predict(data: Data, model: SeqLabel, text: List[List[str]]) -> NERResponse:
    with NamedTemporary() as tmpfile:
        with open(tmpfile, 'w') as tmpf:
            for sent in text:
                tmpf.write('\n'.join(map(lambda x: x + '\tO',sent)))
                tmpf.write('\n\n')
        
        data.raw_dir = tmpfile
        data.generate_instance('raw')
        speed, pred_results, pred_scores= evaluate(data, model, 'raw', skip_eval=True) # type: ignore
        
    print(pred_results)
        
    prediction_result = []
    for row1, row2 in zip(text, pred_results): 
        tokens_labels = [NERLabelledToken(token=t, label=l) for t, l in zip(row1, row2)]
        prediction_result.append(tokens_labels)

    return NERResponse(
        prediction=prediction_result
    )
    
    
def pred_result_to_df(text: List[List[str]]):
    pass
    
async def hybrid_predict(text: List[List[str]]) -> NERResponse:
    if ModelEnum.hybrid not in models or ModelEnum.morph not in models:
        raise HTTPException(status_code=404, detail="The requisite models were not properly loaded by the API.")
    
    
    ma = yap.yap_ma_api(ner.raw_toks_str_from_ner_df(tok))
    
    print("MA complete")
    
    x = prune_lattices(ma, multi)
    
    print("Pruning complete")
    
    
    print("Gold morph + fallback")

    ma = yap.yap_ma_api(ner.raw_toks_str_from_ner_df(tok))
    
    print(ma)
    
    pruned = prune_lattices(ma, multi, fallback=True)
    
    md = yap.yap_joint_from_lattice_api(pruned)
    
    md['TOKEN'] = md['TOKEN'].astype(str)
    
    with open('utils_eval_files/yap_hybrid_gold_multi_fallback_dev_tokens.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['TOKEN'].agg('\n'.join)))
    
    md['FORM'] = md['FORM'].apply(lambda x: x + ' O')

    with open('utils_eval_files/yap_hybrid_gold_multi_fallback_dev.txt', 'w') as w:
        w.write('\n\n'.join(md.groupby('SENTNUM')['FORM'].agg('\n'.join)))
    


@app.post('/predict')
async def api_predict(q: NERQuery) -> NERResponse:
    tt = tokenize_sentences(text2listOfSentences(q.text))
    if q.model == ModelEnum.hybrid:
        return await hybrid_predict(tt)
    
    if q.model not in models:
        raise HTTPException(status_code=404, detail=f"The model '{q.model}' has not been loaded, please try one of {list(models.keys())}")
    data, model = models[q.model]
    
    return standard_predict(data, model, tt)
    


@app.get('/healthcheck')
def health():
    return "OK"




if __name__ == '__main__':
    uvicorn.run('ner_app:app', port=5000, reload=True)