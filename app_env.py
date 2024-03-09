import os
from dotenv import load_dotenv


load_dotenv()

class ENV:
    ABSOLUTE_PATH_HEBREW_NER=os.environ.get('ABSOLUTE_PATH_HEBREW_NER')
    CORPUS_DIR=os.environ.get('CORPUS_DIR')