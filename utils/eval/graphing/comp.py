from matplotlib import pyplot as plt
import matplotlib.style as style
import numpy as np
from utils.eval import eval_morph_ftam, eval_multi, eval_single, eval_morph, eval_trn_morph, eval_trn_multi, eval_trn_single
from utils.eval.conf_interval import norm_approx_int
from matplotlib import rc
from utils.ner import read_file_to_sentences_df
from config import DEV, TEST

# activate latex text rendering
rc('text', usetex=True)
style.use('seaborn-v0_8-colorblind')

DEV_NUM_TOKS = read_file_to_sentences_df(DEV.TOK)
DEV_NUM_SENTS = max(read_file_to_sentences_df(DEV.TOK)['SentNum'])
CONFIDENCE = 0.95

def base_comp_graph(categories, orig_values, orig_label, compared_values, compared_label, orig_yerr=None, comp_yerr=None,
                    x_label='', y_label='', title='', bar_width=0.2, dpi=800, save=None):
    bar_width = bar_width * len(categories)
    
    orig_positions = np.arange(len(categories))
    compared_positions = orig_positions + bar_width
    
    fig, ax = plt.subplots()
    
    orig_bars = ax.bar(orig_positions, orig_values, bar_width, 
                       yerr=orig_yerr,
                       label=orig_label)
    compared_bars = ax.bar(compared_positions, compared_values, bar_width,
                           yerr=comp_yerr,
                           label=compared_label)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    ax.set_ylim(0, 100) # percentage scores
    ax.set_xticks(orig_positions + bar_width / 2, 
                  labels=categories)
    
    bar_label_fmt = "$%0.2f$"
    ax.bar_label(orig_bars, padding=0.5, fmt=bar_label_fmt)
    ax.bar_label(compared_bars, padding=0.5, fmt=bar_label_fmt)
    
    ax.grid(axis='y')
    ax.legend(loc='upper left', fontsize='small')
    
    if save is not None: 
        plt.savefig(save, dpi=dpi)
        
    return fig
        

def basic_on_dev():
    gold_morph = eval_morph_ftam.eval_morph_ftam_dev()
    tok = eval_single.eval_single_dev()
    multi = eval_multi.eval_multi_dev()

    categories = ['Token-Single', 'Token-Multi', 'Morpheme']
    orig_values = np.array([78.15,77.59,80.30])
    my_values = np.array([tok.f, multi.f,  gold_morph.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$', my_values * 100, 'Recreated',
                    x_label='NER Type', y_label='F1 Scores (token-level evaluation)',
                    title='Comparison between reported results and my recreated results - dev\n' +r'\small{token-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, DEV_NUM_SENTS) * 100 for f1 in my_values],
                    orig_yerr=[0.3, 0.4, 0.5],
                    # save=None,
                    save='graphs/standard/token_eval_dev.png',
                    bar_width=0.08)
    
def basic_on_test():
    gold_morph = eval_morph_ftam.eval_morph_ftam_test()
    tok = eval_single.eval_single_test()
    multi = eval_multi.eval_multi_test()

    categories = ['Token-Single', 'Token-Multi', 'Morpheme']
    orig_values = np.array([77.15,77.75,79.28])
    my_values = np.array([tok.f, multi.f,  gold_morph.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$', my_values * 100, 'Recreated',
                    x_label='NER Type', y_label='F1 Scores (token-level evaluation)',
                    title='Comparison between reported results and my recreated results - test\n' +r'\small{token-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, DEV_NUM_SENTS) * 100 for f1 in my_values],
                    orig_yerr=[0.6, 0.3, 0.7],
                    # save=None,
                    save='graphs/standard/token_eval_test.png',
                    bar_width=0.08)
    

def morph_on_dev():
    _, gold_morph, pure_yap, pred_multi, gold_multi = eval_morph_ftam.eval_all_morph_ftam_dev()

    categories = ['Gold', 'Pure Yap', 'Hybrid - Pred Multi', 'Hybrid - Gold Multi']
    orig_values = np.array([80.30,74.52,79.04,79.04])
    my_values = np.array([gold_morph.f, pure_yap.f,  pred_multi.f, gold_multi.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$', my_values * 100, 'Recreated',
                    x_label='Morpheme Model Type', y_label='F1 Scores (normalised morpheme-level evaluation)',
                    title='Comparison between reported results and my recreated results - dev\n' +r'\small{normalised morpheme-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, DEV_NUM_SENTS) * 100 for f1 in my_values],
                    orig_yerr=[0.4, 0.5, 0.5, 0.5],
                    # save=None,
                    save='graphs/standard/morph_eval_dev.png',
                    bar_width=0.08)
    
def morph_on_test():
    _, gold_morph, pure_yap, pred_multi, gold_multi = eval_morph_ftam.eval_all_morph_ftam_test()

    categories = ['Gold', 'Pure Yap', 'Hybrid - Pred Multi', 'Hybrid - Gold Multi']
    orig_values = np.array([79.10,69.52,77.11,77.11])
    my_values = np.array([gold_morph.f, pure_yap.f,  pred_multi.f, gold_multi.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$', my_values * 100, 'Recreated',
                    x_label='Morpheme Model Type', y_label='F1 Scores (normalised morpheme-level evaluation)',
                    title='Comparison between reported results and my recreated results - test\n' +r'\small{normalised morpheme-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, DEV_NUM_SENTS) * 100 for f1 in my_values],
                    orig_yerr=[0.6, 0.6, 0.7, 0.7],
                    # save=None,
                    save='graphs/standard/morph_eval_test.png',
                    bar_width=0.08)
  
    
def basic_trn_on_dev():
    gold_morph = eval_trn_morph.eval_trn_morph_dev()
    tok = eval_trn_single.eval_single_dev()
    multi = eval_trn_multi.eval_multi_dev()

    categories = ['Token-Single', 'Token-Multi', 'Morpheme']
    orig_values = np.array([78.15,77.59,80.30])
    my_values = np.array([tok.f, multi.f,  gold_morph.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$', my_values * 100, 'Transformer Labeller',
                    x_label='NER Type', y_label='F1 Scores (token-level evaluation)',
                    title="Comparison between NEMO$^2$ results and my novel architecture's results - dev\n" +r'\small{token-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, DEV_NUM_SENTS) * 100 for f1 in my_values],
                    orig_yerr=[0.3, 0.4, 0.5],
                    # save=None,
                    save='graphs/transformer/token_eval_dev.png',
                    bar_width=0.08)
    
    
def basic_trn_on_test():
    gold_morph = eval_trn_morph.eval_trn_morph_test()
    tok = eval_trn_single.eval_single_test()
    multi = eval_trn_multi.eval_multi_test()

    categories = ['Token-Single', 'Token-Multi', 'Morpheme']
    orig_values = np.array([77.15,77.75,79.28])
    my_values = np.array([tok.f, multi.f,  gold_morph.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$', my_values * 100, 'Transformer Labeller',
                    x_label='NER Type', y_label='F1 Scores (token-level evaluation)',
                    title="Comparison between NEMO$^2$ results and my novel architecture's results - test\n" +r'\small{token-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, DEV_NUM_SENTS) * 100 for f1 in my_values],
                    orig_yerr=[0.6, 0.3, 0.7],
                    # save=None,
                    save='graphs/transformer/token_eval_test.png',
                    bar_width=0.08)


def morph_trn_on_dev():
    _, gold_morph, pure_yap, pred_multi, gold_multi = eval_trn_morph.eval_all_trn_morph_dev()

    categories = ['Gold', 'Pure Yap', 'Hybrid - Pred Multi', 'Hybrid - Gold Multi']
    orig_values = np.array([80.30,74.52,79.04,79.04])
    my_values = np.array([gold_morph.f, pure_yap.f,  pred_multi.f, gold_multi.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$', my_values * 100, 'Transformer Labeller',
                    x_label='Morpheme Model Type', y_label='F1 Scores (normalised morpheme-level evaluation)',
                    title="Comparison between NEMO$^2$ results and my novel architecture's results - dev\n" +r'\small{normalised morpheme-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, DEV_NUM_SENTS) * 100 for f1 in my_values],
                    orig_yerr=[0.4, 0.5, 0.5, 0.5],
                    # save=None,
                    save='graphs/transformer/morph_eval_dev.png',
                    bar_width=0.08)
    
def morph_trn_on_test():
    _, gold_morph, pure_yap, pred_multi, gold_multi = eval_trn_morph.eval_all_trn_morph_test()

    categories = ['Gold', 'Pure Yap', 'Hybrid - Pred Multi', 'Hybrid - Gold Multi']
    orig_values = np.array([80.30,74.52,79.04,79.04])
    my_values = np.array([gold_morph.f, pure_yap.f,  pred_multi.f, gold_multi.f])
    # my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$', my_values * 100, 'Transformer Labeller',
                    x_label='Morpheme Model Type', y_label='F1 Scores (normalised morpheme-level evaluation)',
                    title="Comparison between NEMO$^2$ results and my novel architecture's results - test\n" +r'\small{normalised morpheme-level evaluation}',
                    comp_yerr=[norm_approx_int(f1, CONFIDENCE, DEV_NUM_SENTS) * 100 for f1 in my_values],
                    orig_yerr=[0.6, 0.6, 0.7, 0.7],
                    # save=None,
                    save='graphs/transformer/morph_eval_test.png',
                    bar_width=0.08)

if __name__ == '__main__':
    basic_on_dev()
    basic_on_test()
    morph_on_dev()
    morph_on_test()
    basic_trn_on_dev()
    basic_trn_on_test()
    morph_trn_on_dev()
    morph_trn_on_test()
    # plt.show()