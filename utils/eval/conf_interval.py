import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import config
import scipy.stats as st
from typing import NamedTuple

from utils import ner

class BootstrapMetric(NamedTuple):
    mean: float
    lower: float
    upper: float
    length: float

def bootstrap(N: int, conf: float, pred: pd.DataFrame, gold: pd.DataFrame):
    SentN = max(pred['SentNum'])
    f1s = []
    merged = pd.merge(pred, gold.drop('Word', axis='columns'), on=['SentNum', 'WordIndex'], how='left', suffixes=['', '_Gold'])
    merged = merged.groupby('SentNum').agg(list)
    for _ in tqdm(range(N), desc='Strapping boots'):
        bootstrap = merged.sample(n=SentN, replace=True)
        f1 = ner.evaluate_token_ner(bootstrap['Label'].explode().to_list(), bootstrap['Label_Gold'].explode().to_list(), verbose=False).f * 100
        f1s.append(f1)
    
    lower, upper = np.percentile(f1s, [50 - conf / 2, 50 + conf / 2])
    mean = np.mean(f1s)
    length = upper - lower
    
    return BootstrapMetric(mean, lower, upper, length) # type: ignore


def norm_approx_int(f1: float, conf: float, N: int):
    """Generates the normal approximation interval at the given N and confidence interval

    Args:
        conf (float): _description_
        N (int): _description_

    Returns:
        _type_: _description_
    """
    z = st.norm.ppf((1 + conf) / 2.0)
    return z * math.sqrt((f1 * (1 - f1)) / N)


if __name__ == '__main__':
    gold = ner.read_file_to_sentences_df(config.TEST.TOK)
    pred = ner.read_file_to_sentences_df('ncrf_results/tok-single/final/test-results.txt')
    
    print(bootstrap(500, 95, pred, gold))