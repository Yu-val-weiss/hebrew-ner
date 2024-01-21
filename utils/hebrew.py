import re

def quotes_to_gershayim(s: str): 
    '''
    Hebrew can sometimes use gershayim, ״ when meaning to use quotes, especially for acronyms.
    
    Yap seems to accept these better.
    '''
    sub = re.compile(r'"')
    return sub.sub('״', s)

def gershayim_to_quotes(s: str):
    '''
    Hebrew can sometimes use gershayim, ״ when meaning to use quotes, especially for acronyms.
    
    This function normalises backs to quotes.
    '''
    sub = re.compile(r'״')
    return sub.sub('"', s)
