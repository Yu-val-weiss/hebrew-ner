from utils import ner


if __name__ == '__main__':
    print('\n\nMulti to multi')

    pred_multi = ner.read_file_to_sentences_df('/Users/yuval/GitHub/hebrew-ner/hpc_eval_results/tok_multi_cnn.txt')
    ner.evaluate_token_ner(pred_multi['Label'].to_list(), multi['Label'].to_list(), multi_tok=True)