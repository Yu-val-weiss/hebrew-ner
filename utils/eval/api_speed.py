import requests
from random import choice
import utils.ner as ner
import config
import statistics
from tqdm import tqdm

# mean: 228.97842693328857ms
# stdev: 223.86241969210678ms   

if __name__ == '__main__':
    tok = ner.read_file_to_sentences_df(config.DEV.TOK)
    
    g = tok.groupby('SentNum')['Word'].agg(list)
    
    
    times = []

    for i in tqdm(range(1000)):
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
        
    
    print('mean:', statistics.mean(times))
    print('stdev:', statistics.stdev(times))