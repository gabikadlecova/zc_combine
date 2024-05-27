import matplotlib.pyplot as plt
import seaborn as sns
import tables_utils as tu

sns.set_style('white')
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'text.latex.preamble': '\\usepackage{times} ', 'figure.figsize': (3.25, 2.0086104634371584), 'figure.constrained_layout.use': True, 'figure.autolayout': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.015, 'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 6, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.titlesize': 8})

#O = Onehot, S = GRAF, F = flops+params, P = ZCP, E = path encoding
def_labels = ['X', 'O', 'S', 'OF', 'SO', 'SF', 'SOF', 'PF', 'POF', 'PSF', 'PSOF']
sel_labels = ['X', 'O', 'S', 'SO', 'PF', 'POF', 'PSF', 'PSOF']

rn = {'val_accs': 'NB201: cifar10', 'imagenet_val_accs': 'NB201: ImageNet16-120', 'darts_val_accs': 'NB301: cifar10'}

flierprops = dict(markersize=5, linestyle='none', marker='o', markerfacecolor='none')

def select_color(column):
    if 'GRAF' in column or 'Everything' == column:
        return sns.color_palette()[0]
    else:
        return sns.color_palette()[1]

def plot_experiments(all_data, targets, train_sizes, include_significance=False, paired=True, file_name=None, labels=def_labels, color=None):
    fig_width = 487.8225/72.27 # \the\textsize / points per inch
    n_ts = len(train_sizes)
    n_tg = len(targets)

    stat_df = None
    if include_significance:
        stat_df = tu.create_stat_table(all_data, paired=True)
    
    plt.subplots(n_tg, n_ts, sharey=True, figsize=(fig_width, n_tg*len(labels)/6.25))
    
    for (si,ts) in enumerate(train_sizes):
        for (ti,target) in enumerate(targets):

            plt_data = all_data[target][ts]
            r_cols = [c for c in labels if c in plt_data.columns]
            plt_data = plt_data[r_cols]
            cols = [tu.FEAT_STRINGS[f].replace('&', '\&') if f in tu.FEAT_STRINGS else f for f in r_cols]
            plt_data.columns=cols

            colors = None
            if color:
                colors = [color for c in cols]
            else:
                colors = [select_color(c) for c in cols]
            
            plt.subplot(n_tg, n_ts, n_ts*ti+si+1)
            ax = sns.boxplot(plt_data, orient='h', palette=colors, linewidth=1, flierprops=flierprops)

            if include_significance:
                s = stat_df[target][ts]
                lpp = len(ax.lines)//len(r_cols)
                ax.lines[:lpp]
                
                import numpy as np
                wh = np.where(s[r_cols] > 0.05)
                
                for l in range(len(r_cols)):
                    if l in np.nditer(wh):
                        for i in range(lpp):
                            ax.lines[l*lpp+i].set_lw(1.5)
                    else:
                        for i in range(lpp):
                            ax.lines[l*lpp+i].set_lw(0.4)
                            
            if si == 0:
                plt.ylabel(f'{rn[target] if target in rn else target}')
            if ti == 0:
                plt.title(f'Training size: {ts}')
    
    if file_name:
        if '.' not in file_name:
            file_name += '.pdf'
        plt.savefig(f'{file_name}')
    plt.show()

def plot_all(all_data, include_significance=False, paired=True, file_name=None, split_targets=None, labels=def_labels, color=None):
    all_data = all_data.iloc[:, all_data.columns.get_level_values(2).isin(labels)]
    ts = list(all_data.columns.get_level_values(0).unique())
    targets = []
    if not split_targets:
        targets = [ts]
    else:
        st = 0
        for s in split_targets:
            targets.append(ts[st:st+s])
            st += s
    
    train_sizes = [32, 128, 1024]
    
    for i,t in enumerate(targets):
        if file_name:
            if '.' not in file_name:
                fn = f'{file_name}-{i+1}'
            else:
                fn = file_name
        else:
            fn = None
        plot_experiments(all_data, t, train_sizes, include_significance=include_significance, paired=paired, file_name=fn, labels=labels, color=color)