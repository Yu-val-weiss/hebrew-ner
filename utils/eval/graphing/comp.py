from matplotlib import pyplot as plt
import matplotlib.style as style
import numpy as np
from utils.eval import eval_morph_ftam, eval_multi, eval_single, eval_morph
from matplotlib import rc

# activate latex text rendering
rc('text', usetex=True)
style.use('seaborn-v0_8-colorblind')


def base_comp_graph(categories, orig_values, orig_label, compared_values, compared_label, x_label='', y_label='', title='', bar_width=0.2, dpi=800, show=True, save=None):
    # where to place the bars
    orig_positions = np.arange(len(categories))
    compared_positions = orig_positions + bar_width
    
    fig, ax = plt.subplots()
    
    orig_bars = ax.bar(orig_positions, orig_values, bar_width, label=orig_label)
    compared_bars = ax.bar(compared_positions, compared_values, bar_width, label=compared_label)
    
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
    
    if show:
        plt.show()
        

if __name__ == '__main__':
    gold_morph = eval_morph_ftam.eval_morph_ftam_dev()
    tok = eval_single.eval_single_dev()
    multi = eval_multi.eval_multi_dev()

    categories = ['Token-Single', 'Token-Multi', 'Morpheme']
    orig_values = np.array([78.15,77.59,80.30])
    my_values = np.array([tok.f, multi.f,  gold_morph.f])
    my_values = my_values * 100

    base_comp_graph(categories, orig_values, 'NEMO$^2$', my_values, 'Recreated',
                    x_label='NER Type', y_label='F1 Scores (token-level evaluation)',
                    title='Comparison between reported results and my recreated results\n' +r"\small{token-level evaluation}",
                    # save=None,
                    save='graphs/standard/token_eval.png',
                    bar_width=0.25)