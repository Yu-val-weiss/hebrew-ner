import os
from dotenv import load_dotenv

load_dotenv(override=True)

class ENV:
    ABSOLUTE_PATH_HEBREW_NER=os.environ.get('ABSOLUTE_PATH_HEBREW_NER')
    CORPUS_DIR=os.environ.get('CORPUS_DIR')
    YAP_HOST=os.environ.get('YAP_HOST')
    YAP_PORT=os.environ.get('YAP_PORT')