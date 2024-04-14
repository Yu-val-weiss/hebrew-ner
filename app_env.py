import os
from dotenv import load_dotenv
from pydantic.dataclasses import dataclass

load_dotenv(override=True, verbose=True)

@dataclass
class ENV:
    ABSOLUTE_PATH_HEBREW_NER: str=os.environ['ABSOLUTE_PATH_HEBREW_NER']
    CORPUS_DIR: str=os.environ['CORPUS_DIR']
    YAP_HOST: str=os.environ['YAP_HOST']
    YAP_PORT: str=os.environ['YAP_PORT']