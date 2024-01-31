import matplotlib
from matplotlib import pyplot as plt
import matplotlib.style as style
import numpy as np
from utils.eval import eval_morph_ftam, eval_multi, eval_single

if __name__ == '__main__':
    style.use('seaborn-v0_8-colorblind')

    (gold_morph, pure_yap, pred_multi, gold_multi) = eval_morph_ftam.eval_morph_ftam()
    tok = eval_single.eval_single()
    multi = eval_multi.eval_multi()
    
    # Data
    categories = ['Token', 'Multi', 'Morph']
    orig_values = np.array([78.15,77.59,80.30])
    mine_values = np.array([tok.f, multi.f,  gold_morph.f])
    mine_values = mine_values * 100

    # Bar width
    bar_width = 0.2

    # Set positions for bar groups
    orig_positions = np.arange(len(categories))
    mine_positions = orig_positions + bar_width

    # Create bar chart
    fig, ax = plt.subplots()

    # Bar for original values
    orig_bars = ax.bar(orig_positions, orig_values, bar_width, label='Original')

    # Bar for mine values
    my_bars = ax.bar(mine_positions, mine_values, bar_width, label='Mine')

    # Set labels, title, and legend
    ax.set_xlabel('Categories')
    ax.set_ylabel('Scores')
    ax.set_ylim(0, 100)
    ax.set_title('Comparison between Mine and Original')
    ax.set_xticks(orig_positions + bar_width / 2)
    ax.set_xticklabels(categories)
    ax.grid(axis='y')
    ax.legend()
    
    
    for bar, barr, orig_value, mine_value in zip(orig_bars, my_bars, orig_values, mine_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f'{orig_value:.2f}', ha='center', va='bottom', color='black', fontsize=8)

        ax.text(barr.get_x() + barr.get_width() / 2, barr.get_height() + 3,
                f'{mine_value:.2f}', ha='center', va='bottom', color='black', fontsize=8)


    # Show the plot
    plt.show()