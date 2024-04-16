from typing import Dict
import requests
from random import choice
import utils.ner as ner
import config
from scipy import stats
import numpy as np
from tqdm import tqdm
import json

def speed_test(N = 500):
    tok = ner.read_file_to_sentences_df(config.DEV.TOK)
    
    g = tok.groupby('SentNum')['Word'].agg(list)
    
    results: Dict = {
        'N': N
    }
    
    times = []
    for _ in tqdm(range(N), desc='Hybrid calls'):
        sent = ' '.join(choice(g))
        r = requests.post(
            'http://127.0.0.1:5000/predict',
            json={
                "text": sent,
                "model": "hybrid",
            }
        )
        
        r.raise_for_status()
        
        times.append(float(r.headers['x-process-time-ms']))
    
        
    print('hybrid')
    print('min,max', np.min(times), np.max(times))
    print('mean:', mean := np.mean(times))
    print('stdev:', std := np.std(times))
    print('conf interval:', conf := stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(N)))
    
    results['hybrid'] = {
        'mean': mean,
        'stdev': std,
        'confidence_interval': conf,
        'plusminus': mean - conf[0],
    }
    
    times = []
    for i in tqdm(range(N), desc='Standard calls'):
        sent = ' '.join(choice(g))
        r = requests.post(
            'http://127.0.0.1:5000/predict',
            json={
                "text": sent,
                "model": "token_single",
            }
        )
        
        r.raise_for_status()
        
        times.append(float(r.headers['x-process-time-ms']))
    
        
    print('token-single')
    print('min,max', np.min(times), np.max(times))
    print('mean:', mean := np.mean(times))
    print('stdev:', std := np.std(times))
    print('conf interval:', conf := stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(N)))
    
    results['token-single'] = {
        'mean': mean,
        'stdev': std,
        'confidence_interval': conf,
        'plusminus': mean - conf[0],
    }
    
    print(results)
    
    with open('utils/eval/api_speed_result.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    speed_test(500)
    