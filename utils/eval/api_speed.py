import requests
from random import choice
import utils.ner as ner
import conf

def make_req(df):
    max_sent_num = max(df['SentNum'])
    rand_sent = choice(list(range(max_sent_num + 1)))
    

if __name__ == '__main__':
    ner.read_file_to_sentences_df(conf.TEST)