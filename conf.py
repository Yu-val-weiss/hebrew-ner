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

@dataclass
class TRAIN:
    MORPH: str = f"{ENV.CORPUS_DIR}/data/spmrl/gold/morph_gold_train.bmes"
    MULTI: str = f"{ENV.CORPUS_DIR}/data/spmrl/gold/token-multi_gold_train.bmes"
    TOK: str = f"{ENV.CORPUS_DIR}/data/spmrl/gold/token-single_gold_train.bmes"

@dataclass
class TEST:
    MORPH: str = f"{ENV.CORPUS_DIR}/data/spmrl/gold/morph_gold_test.bmes"
    MULTI: str = f"{ENV.CORPUS_DIR}/data/spmrl/gold/token-multi_gold_test.bmes"
    TOK: str = f"{ENV.CORPUS_DIR}/data/spmrl/gold/token-single_gold_test.bmes"

@dataclass
class DEV:
    MORPH: str = f"{ENV.CORPUS_DIR}/data/spmrl/gold/morph_gold_dev.bmes"
    MULTI: str = f"{ENV.CORPUS_DIR}/data/spmrl/gold/token-multi_gold_dev.bmes"
    TOK: str = f"{ENV.CORPUS_DIR}/data/spmrl/gold/token-single_gold_dev.bmes"