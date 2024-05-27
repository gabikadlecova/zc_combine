import pandas as pd
from scipy import stats
import scikit_posthocs as ph
import matplotlib.pyplot as plt
import seaborn as sns

FEAT_STRINGS = {'X': 'None (BRP-NAS)', 
                'O': 'OH', 
                'S': 'GRAF', 
                'F': 'FP',
                'OF': 'OH + FP', 
                'SO': 'GRAF + OH', 
                'SF': 'GRAF + FP', 
                'SOF': 'GRAF + OH + FP', 
                'PF': 'ZCP', 
                'POF': 'ZCP + OH', 
                'PSF': 'ZCP + GRAF', 
                'PSOF': 'ZCP + GRAF + OH',
                'XE': 'PE',
                'OE': 'OH + PE', 
                'SE': 'GRAF + PE', 
                'OFE': 'OH + FP + PE', 
                'SOE': 'GRAF + OH + PE', 
                'SFE': 'GRAF + FP + PE', 
                'SOFE': 'GRAF + OH + FP + PE', 
                'PFE': 'ZCP + PE', 
                'POFE': 'ZCP + OH + PE', 
                'PSFE': 'ZCP + GRAF + PE', 
                'PSOFE': 'ZCP + GRAF + OH + PE'
               }

CONFIG_TRAINSIZE = {'config32': 32, 'config128_16': 128, 'config1024_16': 1024}

base_labels = ['X', 'O', 'OF', 'S', 'SF', 'SO', 'SOF', 'PF', 'POF', 'PSF', 'PSOF']
labels = base_labels
labels += [bl + 'E' for bl in base_labels]

labels += [bl + 'W' for bl in base_labels]
for b in [bl + 'W' for bl in base_labels]:
    FEAT_STRINGS[b] = FEAT_STRINGS[b[:-1]] + ' + WL'

labels += [bl + 'A' for bl in base_labels]
for b in [bl + 'A' for bl in base_labels]:
    FEAT_STRINGS[b] = FEAT_STRINGS[b[:-1]] + ' + A2V'

labels += [bl + 'M' for bl in base_labels]
for b in [bl + 'M' for bl in base_labels]:
    FEAT_STRINGS[b] = FEAT_STRINGS[b[:-1]] + ' (MO)'


def latex_table(df, stats, columns=None, caption='', hlines=[0], labels=labels):
    table_mean = df.mean().unstack()
    table_std = df.std().unstack()
    table_stat = stats.unstack()

    cols = [c for c in labels if c in table_mean.columns]

    table_mean = table_mean[cols]
    table_std = table_std[cols]
    table_stat = table_stat[cols]
    cols = [FEAT_STRINGS[f] if f in FEAT_STRINGS else f for f in cols]
    table_mean.columns = cols
    table_std.columns = cols
    table_stat.columns = cols

    table = pd.merge(table_mean, table_std, left_index=True, right_index=True, suffixes=('_mean', '_std'))
    table = pd.merge(table, table_stat, left_index=True, right_index=True, suffixes=('', '_stat'))
    
    def pm_formatter(x):
        if x[2] >= 0.05:
            return '$\mathbf{'+f'{x[0]:.2f}' + '^{' + f'{x[1]:.2f}' + '}}$'
        return f'${x[0]:.2f}' + '^{' + f'{x[1]:.2f}' + '}$'
    
    table_str = pd.DataFrame()
    for c in cols:
        table_str[c] = table[[f'{c}_mean', f'{c}_std', f'{c}']].apply(tuple, axis=1)
    table_str = table_str.T
    
    #columns are now different targets and different train size, rows are different feature sets
    
    start="""\\begin{table}
    \\small
    \\addtolength{\\tabcolsep}{-0.2em}
    \\centering
    \\caption{%s}
    \\vskip 0.15in
    """
    
    end="""
    \\end{table}"""

    out = table_str[columns].style.format(formatter=pm_formatter).to_latex(column_format='r' + len(columns)*'|ccc', multicol_align='|c')
    out = out.replace('_', '\_')
    out = out.splitlines()
    for (i,h) in enumerate(hlines):
        out[h+3+i:h+3+i] = ['\\hline']
    res = start % caption
    res += '\n'.join(out)
    res += end

    return res

def create_stat_table(df, paired):
    stat_df = df.mean()
    
    targets = df.columns.get_level_values(0).unique()
    for t in targets:
        train_sizes = df[t].columns.get_level_values(0).unique()
        for s in train_sizes:
        
            td = df[t][s]
            pval = None
            if paired:
                pval = stats.friedmanchisquare(*td.T.values).pvalue
            else: 
                pval = stats.kruskal(*td.T.values).pvalue
            
            if pval > 0.05:
                print(f'All ranks may be equally distributed: Friedman test p-value={pval}')
            
            methods = td.columns
            best_method = td.median().argmax()
            best_method = td.columns[best_method]
            
            tf = td.unstack().reset_index(0)
            tf.columns=['group', 'value']
            
            ph_res = None

            if paired:
                ph_res = ph.posthoc_wilcoxon(tf, group_col='group', val_col='value', p_adjust='holm')
            else:
                ph_res = ph.posthoc_mannwhitney(tf, group_col='group', val_col='value', p_adjust='holm')
    
            hg = []
            for m in methods:
                stat_df.loc[(t,s,m)]=ph_res[best_method][m].item()
    
    return stat_df
    
def feat_string(row):
    out = ''
    if row['use_all_proxies']:
        out += 'P'
    if row['use_features']:
        out += 'S'
    if row['use_onehot']:
        out += 'O'
    if row['use_flops_params'] or row['use_all_proxies']:
        out += 'F'
    if row['use_path_encoding']:
        out += 'E'
    if 'use_wl_embedding' in row.index and row['use_wl_embedding']:
        out += 'W'
    if 'use_embedding' in row.index and row['use_embedding']:
        out += 'A'
    if 'multi_objective' in row.index and row['multi_objective']:
        out += 'M'
    if out == '':
        out = 'X'
        
    return out

def load_data(file_name):
    
    data = pd.read_csv(file_name)
    if 'use_path_encoding' not in data.columns:
        data['use_path_encoding']=False
    data['use_path_encoding'].fillna(False, inplace=True)
    
    data['features'] = data.apply(feat_string, axis=1)
    
    all_data = data[['dataset', 'train_size', 'features', 'data_seed', 'tau']]
    all_data = all_data.set_index(['dataset', 'train_size', 'features', 'data_seed'])
    
    all_data = all_data.unstack(3).T
    all_data.index = all_data.index.get_level_values(1)
    
    return all_data

def load_wandbd_dump(file_name):
    import pickle
    if not isinstance(file_name, list):
        file_name = [file_name]

    dfs = [pickle.load(open(f, 'rb')) for f in file_name]
    runs_df = pd.concat(dfs)
    
    exp_groups = [r['config']['exp_group'] for _, r in runs_df.iterrows()]
    targets = [eg[:eg.find('_gcn')] for eg in set(exp_groups)]
    targets = set(targets)
    
    from collections import defaultdict, Counter
    
    configs={'config1024_16', 'config128_16', 'config32'}
    
    runs_groups = defaultdict(list)
    groups_runids = defaultdict(list)
    groups_rowids = defaultdict(list)
    augments_groups = dict()
    
    for i,r in runs_df.iterrows():
        run_gr = r['config']['exp_group']
        run_id = r['config']['runid']
        augments = r['config']['foresight_augment']
        runs_groups[run_gr].append(run_id)
        groups_runids[run_gr].append(r['id'])
        groups_rowids[run_gr].append(i)
        augments_groups[run_gr]=augments

    all_data = dict()
    
    group_taus = defaultdict(list)

    for i,r in runs_df.iterrows():
        if 'tau' not in r['summary']:
            continue
        group_taus[r['config']['exp_group']].append(r['summary']['tau'])
    
    for t in targets:
        for c in configs:
            values = []
            found_labels = set()
            c_labels = []
            for l in labels:
                for k,v in group_taus.items():
                    if k.startswith(t) and c in k and f'_{l}_' in k:
                        found_labels |= {l}
                        values.append(v)
                        c_labels.append(l)
            data=pd.DataFrame(values).T
            data.columns=c_labels
            for c1 in c_labels:
                all_data[(f'{t}', CONFIG_TRAINSIZE[c], c1)]=data[c1]
            
    all_data=pd.DataFrame(all_data)
    all_data.columns.set_names(['dataset', 'train_size', 'features'], inplace=True)
    return all_data