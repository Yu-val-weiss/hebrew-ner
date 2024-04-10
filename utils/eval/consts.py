from app_env import ENV

class DEV:
    MORPH = f"{ENV.CORPUS_DIR}/data/spmrl/gold/morph_gold_dev.bmes"
    MULTI = f"{ENV.CORPUS_DIR}/data/spmrl/gold/token-multi_gold_dev.bmes"
    TOK = f"{ENV.CORPUS_DIR}/data/spmrl/gold/token-single_gold_dev.bmes"
    
class TEST:
    MORPH = f"{ENV.CORPUS_DIR}/data/spmrl/gold/morph_gold_test.bmes"
    MULTI = f"{ENV.CORPUS_DIR}/data/spmrl/gold/token-multi_gold_test.bmes"
    TOK = f"{ENV.CORPUS_DIR}/data/spmrl/gold/token-single_gold_test.bmes"