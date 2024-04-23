from matplotlib import pyplot as plt
import matplotlib.style as style
import numpy as np
from utils.eval import eval_morph_ftam, eval_multi, eval_single, eval_morph, eval_trn_morph, eval_trn_multi, eval_trn_single
from utils.eval.conf_interval import norm_approx_int
from matplotlib import rc
from utils.ner import read_file_to_sentences_df, make_spans
from config import DEV, TEST

# activate latex text rendering
rc('text', usetex=True)
style.use('seaborn-v0_8-colorblind')

DEV_TOK = read_file_to_sentences_df(DEV.TOK)
DEV_NUM_TOKS = len(DEV_TOK)
DEV_NUM_SENTS = max(DEV_TOK['SentNum'])
DEV_NUM_SPANS = len(make_spans(DEV_TOK['Label']))

TEST_TOK = read_file_to_sentences_df(TEST.TOK)
TEST_NUM_TOKS = len(TEST_TOK)
TEST_NUM_SENTS = max(TEST_TOK['SentNum'])
TEST_NUM_SPANS = len(make_spans(TEST_TOK['Label']))

CONFIDENCE = 0.95

def base_comp_graph(categories, orig_values, orig_label, compared_values, compared_label, orig_yerr=None, comp_yerr=None,
                    x_label='', y_label='', title='', bar_width=0.2, dpi=800, save=None, figax=None):
    bar_width = bar_width * len(categories)
    
    orig_positions = np.arange(len(categories))
    compared_positions = orig_positions + bar_width
    
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    
    orig_bars = ax.bar(orig_positions, orig_values, bar_width, 
                       yerr=orig_yerr,
                       label=orig_label)
    compared_bars = ax.bar(compared_positions, compared_values, bar_width,
                           yerr=comp_yerr,
                           label=compared_label)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    ax.set_ylim(60, 90) # percentage scores
    ax.set_xticks(orig_positions + bar_width / 2, 
                  labels=categories)
    
    bar_label_fmt = "$%0.2f$"
    ax.bar_label(orig_bars, padding=0.5, fmt=bar_label_fmt)
    ax.bar_label(compared_bars, padding=0.5, fmt=bar_label_fmt)
    
    ax.grid(axis='y')
    ax.legend(loc='upper left', fontsize='small')
    
    if save is not None: 
        fig.savefig(save, dpi=dpi)
        
    return fig
        

def basic_on_dev(save='graphs/standard/token_eval_dev.png', figax=None):
    gold_morph = eval_morph_ftam.eval_morph_ftam_dev()
    tok = eval_single.eval_single_dev()
    multi = eval_multi.eval_multi_dev()

    categories = ['Token-Single', 'Token-Multi', 'Morpheme']
    orig_values = np.array([78.15,77.59,80.30])
    my_values = np.array([tok.f, multi.f,  gold_morph.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$ (Bareket and Tsarfaty, 2021)', my_values * 100, 'This Work (My Core Model)',
                    x_label='NER Type', y_label='F1 Scores (token-level evaluation)',
                    title='Comparison between reported results and my recreated results - dev\n' +r'\small{token-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, DEV_NUM_SPANS) * 100 for f1 in my_values],
                    orig_yerr=[0.3, 0.4, 0.5],
                    # save=None,
                    save=save,
                    figax=figax,
                    bar_width=0.08)
    
def basic_on_test(save='graphs/standard/token_eval_test.png', figax=None):
    gold_morph = eval_morph_ftam.eval_morph_ftam_test()
    tok = eval_single.eval_single_test()
    multi = eval_multi.eval_multi_test()

    categories = ['Token-Single', 'Token-Multi', 'Morpheme']
    orig_values = np.array([77.15,77.75,79.28])
    my_values = np.array([tok.f, multi.f,  gold_morph.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$ (Bareket and Tsarfaty, 2021)', my_values * 100, 'This Work (My Core Model)',
                    x_label='NER Type', y_label='F1 Scores (token-level evaluation)',
                    title='Comparison between reported results and my recreated results - test\n' +r'\small{token-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, TEST_NUM_SPANS) * 100 for f1 in my_values],
                    orig_yerr=[0.6, 0.3, 0.6],
                    # save=None,
                    save=save,
                    figax=figax,
                    bar_width=0.08)
    

def morph_on_dev(save='graphs/standard/morph_eval_dev.png', figax=None):
    _, gold_morph, pure_yap, pred_multi, gold_multi = eval_morph_ftam.eval_all_morph_ftam_dev()

    categories = ['Gold', 'Pure YAP', 'Hybrid - Pred Multi', 'Hybrid - Gold Multi']
    orig_values = np.array([80.30,74.52,79.04,79.04])
    my_values = np.array([gold_morph.f, pure_yap.f,  pred_multi.f, gold_multi.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$ (Bareket and Tsarfaty, 2021)', my_values * 100, 'This Work (My Core Model)',
                    x_label='Morpheme Model Type', y_label='F1 Scores (normalised morpheme-level evaluation)',
                    title='Comparison between reported results and my recreated results - dev\n' +r'\small{normalised morpheme-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, DEV_NUM_SPANS) * 100 for f1 in my_values],
                    orig_yerr=[0.5, 0.7, 0.5, 0.5],
                    # save=None,
                    save=save,
                    figax=figax,
                    bar_width=0.08)
    
def morph_on_test(save='graphs/standard/morph_eval_test.png', figax=None):
    _, gold_morph, pure_yap, pred_multi, gold_multi = eval_morph_ftam.eval_all_morph_ftam_test()

    categories = ['Gold', 'Pure YAP', 'Hybrid - Pred Multi', 'Hybrid - Gold Multi']
    orig_values = np.array([79.28,73.53,77.64,77.64])
    my_values = np.array([gold_morph.f, pure_yap.f,  pred_multi.f, gold_multi.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$ (Bareket and Tsarfaty, 2021)', my_values * 100, 'This Work (My Core Model)',
                    x_label='Morpheme Model Type', y_label='F1 Scores (normalised morpheme-level evaluation)',
                    title='Comparison between reported results and my recreated results - test\n' +r'\small{normalised morpheme-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, TEST_NUM_SPANS) * 100 for f1 in my_values],
                    orig_yerr=[0.6, 0.8, 0.7, 0.7],
                    # save=None,
                    save=save,
                    figax=figax,
                    bar_width=0.08)
    
    
def std_dev():
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(16,5))
    
    # basic
    basic_on_dev(save=None, figax=(fig, ax0))
    
    # morph
    morph_on_dev(save=None, figax=(fig,ax1))
    
    fig.savefig('graphs/standard/dev_comb.png', dpi=800, bbox_inches='tight')
    

def std_test():
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(16,5))
    
    # basic
    basic_on_test(save=None, figax=(fig, ax0))
    
    # morph
    morph_on_test(save=None, figax=(fig,ax1))
    
    fig.savefig('graphs/standard/test_comb.png', dpi=800, bbox_inches='tight')
    
def basic_trn_on_dev(save='graphs/transformer/token_eval_dev.png', figax=None):
    gold_morph = eval_trn_morph.eval_trn_morph_dev()
    tok = eval_trn_single.eval_single_dev()
    multi = eval_trn_multi.eval_multi_dev()

    categories = ['Token-Single', 'Token-Multi', 'Morpheme']
    orig_values = np.array([78.15,77.59,80.30])
    my_values = np.array([tok.f, multi.f,  gold_morph.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$ (Bareket and Tsarfaty, 2021)', my_values * 100, 'This Work (My Transformer Model)',
                    x_label='NER Type', y_label='F1 Scores (token-level evaluation)',
                    title="Comparison between NEMO$^2$ results and my novel architecture's results - dev\n" +r'\small{token-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, DEV_NUM_SPANS) * 100 for f1 in my_values],
                    orig_yerr=[0.3, 0.4, 0.5],
                    # save=None,
                    save=save,
                    figax=figax,
                    bar_width=0.08)
    
    
def basic_trn_on_test(save='graphs/transformer/token_eval_test.png', figax=None):
    gold_morph = eval_trn_morph.eval_trn_morph_test()
    tok = eval_trn_single.eval_single_test()
    multi = eval_trn_multi.eval_multi_test()

    categories = ['Token-Single', 'Token-Multi', 'Morpheme']
    orig_values = np.array([77.15,77.75,79.28])
    my_values = np.array([tok.f, multi.f,  gold_morph.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$ (Bareket and Tsarfaty, 2021)', my_values * 100, 'This Work (My Transformer Model)',
                    x_label='NER Type', y_label='F1 Scores (token-level evaluation)',
                    title="Comparison between NEMO$^2$ results and my novel architecture's results - test\n" +r'\small{token-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, TEST_NUM_SPANS) * 100 for f1 in my_values],
                    orig_yerr=[0.6, 0.3, 0.6],
                    # save=None,
                    save=save,
                    figax=figax,
                    bar_width=0.08)


def morph_trn_on_dev(save='graphs/transformer/morph_eval_dev.png', figax=None):
    _, gold_morph, pure_yap, pred_multi, gold_multi = eval_trn_morph.eval_all_trn_morph_dev()

    categories = ['Gold', 'Pure YAP', 'Hybrid - Pred Multi', 'Hybrid - Gold Multi']
    orig_values = np.array([80.30,74.52,79.04,79.04])
    my_values = np.array([gold_morph.f, pure_yap.f,  pred_multi.f, gold_multi.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$ (Bareket and Tsarfaty, 2021)', my_values * 100, 'This Work (My Transformer Model)',
                    x_label='Morpheme Model Type', y_label='F1 Scores (normalised morpheme-level evaluation)',
                    title="Comparison between NEMO$^2$ results and my novel architecture's results - dev\n" +r'\small{normalised morpheme-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, DEV_NUM_SPANS) * 100 for f1 in my_values],
                    orig_yerr=[0.5, 0.7, 0.5, 0.5],
                    # save=None,
                    save=save,
                    figax=figax,
                    bar_width=0.08)
    
def morph_trn_on_test(save='graphs/transformer/morph_eval_test.png', figax=None):
    _, gold_morph, pure_yap, pred_multi, gold_multi = eval_trn_morph.eval_all_trn_morph_test()

    categories = ['Gold', 'Pure YAP', 'Hybrid - Pred Multi', 'Hybrid - Gold Multi']
    orig_values = np.array([79.28,73.53,77.64,77.64])
    my_values = np.array([gold_morph.f, pure_yap.f,  pred_multi.f, gold_multi.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$ (Bareket and Tsarfaty, 2021)', my_values * 100, 'This Work (My Transformer Model)',
                    x_label='Morpheme Model Type', y_label='F1 Scores (normalised morpheme-level evaluation)',
                    title="Comparison between NEMO$^2$ results and my novel architecture's results - test\n" +r'\small{normalised morpheme-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, TEST_NUM_SPANS) * 100 for f1 in my_values],
                    orig_yerr=[0.6, 0.8, 0.7, 0.7],
                    # save=None,
                    save=save,
                    figax=figax,
                    bar_width=0.08)
    
    
def trn_dev():
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(16,5))
    
    # basic
    basic_trn_on_dev(save=None, figax=(fig, ax0))
    
    # morph
    morph_trn_on_dev(save=None, figax=(fig,ax1))
    
    fig.savefig('graphs/transformer/dev_comb.png', dpi=800, bbox_inches='tight')
    

def trn_test():
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(16,5))
    
    # basic
    basic_trn_on_test(save=None, figax=(fig, ax0))
    
    # morph
    morph_trn_on_test(save=None, figax=(fig,ax1))
    
    fig.savefig('graphs/transformer/test_comb.png', dpi=800, bbox_inches='tight')
    
    
def std_test_all_in_one():
    gold_morph = eval_morph_ftam.eval_morph_ftam_test()
    tok = eval_single.eval_single_test()
    multi = eval_multi.eval_multi_test()

    categories = ['Token-Single', 'Token-Multi', 'Morpheme (Gold)']
    orig_values = [77.15,77.75,79.28]
    my_values = [tok.f, multi.f,  gold_morph.f]
    
    _, _, pure_yap, pred_multi, gold_multi = eval_morph_ftam.eval_all_morph_ftam_test()

    categories += ['Pure YAP', 'Hybrid - Pred Multi', 'Hybrid - Gold Multi']
    orig_values += [73.53,77.64,77.64]
    my_values += [pure_yap.f,  pred_multi.f, gold_multi.f]
    
    orig_values = np.array(orig_values)
    my_values = np.array(my_values)
    
    figax = plt.subplots(1, 1, figsize=(10,6))
    
    base_comp_graph(categories, orig_values, 'NEMO$^2$ (Bareket and Tsarfaty, 2021)', my_values * 100, 'This Work (My Transformer Model)',
                    x_label='NER Model Type', y_label=r'F1 (\%)',
                    title="Comparison between NEMO$^2$ results and my recreated results - test\n" +r'\small{token-level evaluation, normalising nonstandard as necessary}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, DEV_NUM_SPANS) * 100 for f1 in my_values],
                    orig_yerr=[0.6, 0.3, 0.6, 0.8, 0.7, 0.7],
                    # save=None,
                    save='graphs/standard/test_all_in_one.png',
                    figax=figax,
                    bar_width=0.07)
    
    
def trn_test_all_in_one():
    gold_morph = eval_trn_morph.eval_trn_morph_test()
    tok = eval_trn_single.eval_single_test()
    multi = eval_trn_multi.eval_multi_test()

    categories = ['Token-Single', 'Token-Multi', 'Morpheme (Gold)']
    orig_values = [77.15,77.75,79.28]
    my_values = [tok.f, multi.f,  gold_morph.f]
    
    _, _, pure_yap, pred_multi, gold_multi = eval_trn_morph.eval_all_trn_morph_test()

    categories += ['Pure YAP', 'Hybrid - Pred Multi', 'Hybrid - Gold Multi']
    orig_values += [73.53,77.64,77.64]
    my_values += [pure_yap.f,  pred_multi.f, gold_multi.f]
    
    orig_values = np.array(orig_values)
    my_values = np.array(my_values)
    
    figax = plt.subplots(1, 1, figsize=(10,6))
    
    base_comp_graph(categories, orig_values, 'NEMO$^2$ (Bareket and Tsarfaty, 2021)', my_values * 100, 'This Work (My Transformer Model)',
                    x_label='NER Model Type', y_label=r'F1 (\%)',
                    title="Comparison between NEMO$^2$ results and my novel architecture's results - test\n" +r'\small{token-level evaluation, normalising nonstandard as necessary}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, TEST_NUM_SPANS) * 100 for f1 in my_values],
                    orig_yerr=[0.6, 0.3, 0.6, 0.8, 0.7, 0.7],
                    # save=None,
                    save='graphs/transformer/test_all_in_one.png',
                    figax=figax,
                    bar_width=0.07)
    
    
    
    

if __name__ == '__main__':
    # basic_on_dev()
    # basic_on_test()
    # morph_on_dev()
    # morph_on_test()
    # basic_trn_on_dev()
    # basic_trn_on_test()
    # morph_trn_on_dev()
    # morph_trn_on_test()
    # std_dev()
    # std_test()
    # trn_dev()
    # trn_test()
    std_test_all_in_one()
    trn_test_all_in_one()
    # plt.show()