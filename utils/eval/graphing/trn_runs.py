import os
from matplotlib import pyplot as plt
import matplotlib.style as style
import numpy as np
from matplotlib import rc
import json
from tqdm import tqdm

# activate latex text rendering
rc('text', usetex=True)
style.use('seaborn-v0_8-colorblind')


def base_run_graph(run_labels, values, x_label='', y_label='', title='', bar_width=0.2, dpi=800, show=True, save=None):
    bar_width = bar_width * len(run_labels)
    
    positions = np.arange(len(run_labels))
    
    _, ax = plt.subplots()
    
    bars = ax.bar(positions, values, bar_width, tick_label=run_labels)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    ax.set_ylim(0, 100) # percentage scores
    
    ax.bar_label(bars, padding=0.5, fmt="$%0.2f$")
    
    ax.grid(axis='y')
    # ax.legend(loc='upper left', fontsize='small')
    
    if save is not None: 
        plt.savefig(save, dpi=dpi)
    
    if show:
        plt.show()
        
        
def parse_file(force_no_show=False):
    with open("utils/eval/graphing/trn_run_conf.json", "r") as f:
        configs = json.load(f)
        
    assert isinstance(configs, list)
    
    print("Configs loaded successfully")
    
    for conf in tqdm(configs):
        conf['values'] = np.array(conf['values']) * 100
        if force_no_show:
            conf['show'] = False
        if conf['save'] is not None:
            save_path = f'graphs/transformer/{conf["save"]}.png'
            dirname = os.path.dirname(save_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            conf['save'] = save_path
        base_run_graph(**conf)
    
    
    
    

if __name__ == '__main__':
    parse_file()