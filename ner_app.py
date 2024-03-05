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
    token_single_lstm_ftam = 'token_single_lstm_ftam'


models: Dict[str, Tuple[Data, SeqLabel]] = {}



@asynccontextmanager
async def lifespan(app: FastAPI):
    # load ML models
    for model in ModelEnum:
        path = os.path.join("api_configs", f"{model}.conf")
        if not os.path.exists(path):
            continue
        print(f"Creating model {model}...")
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

        models['token_single'] = (data, model)
    
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
        self.fd, self.fp = tempfile.mkstemp(text=True)
        return self.fp
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.close(self.fd)


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

@app.post('/predict')
def api_predict(q: NERQuery) -> NERResponse:
    if q.model not in models:
        raise HTTPException(status_code=404, detail=f"The model '{q.model}' has not been loaded, please try one of {list(models.keys())}")
    data, model = models[q.model]
    
    tt = tokenize_sentences(text2listOfSentences(q.text))
    with NamedTemporary() as tmpfile:
        with open(tmpfile, 'w') as tmpf:
            for sent in tt:
                tmpf.write('\n'.join(map(lambda x: x + '\tO',sent)))
                tmpf.write('\n\n')
        
        f = open(tmpfile, 'r')

        file_contents = f.read()

        print (file_contents)
        
        f.close()
        
        data.raw_dir = tmpfile
        data.generate_instance('raw')
        speed, pred_results, pred_scores= evaluate(data, model, 'raw', skip_eval=True) # type: ignore
        
    prediction_result = []
    for row1, row2 in zip(tt, pred_results): #type: ignore
        tokens_labels = [NERLabelledToken(token=t, label=l) for t, l in zip(row1, row2)]
        prediction_result.append(tokens_labels)

    return NERResponse(
        prediction=prediction_result
    )

@app.get('/healthcheck')
def health():
    return "OK"




if __name__ == '__main__':
    uvicorn.run('ner_app:app', port=5000, reload=True)