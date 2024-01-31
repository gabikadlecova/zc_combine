from zc_combine.utils.script_utils import create_cache_filename
from zc_combine.utils.script_utils import load_feature_proxy_dataset



from zc_combine.fixes.operations import get_ops_edges_nb201, get_ops_edges_tnb101, get_ops_nb101, get_ops_nb301


def create_map(bench):
    suffix = '_full' if bench != 'nb101' and 'macro' not in bench else '_first'
    cfg = f'../zc_combine/configs/{bench}{suffix}.json'
    dataset = 'cifar10' if 'tnb101' not in bench else 'class_scene'

    version_key = 'paper'
    cache_path = create_cache_filename('../scripts/cache_data/', cfg, None, version_key, True)
    
    _, data, y = load_feature_proxy_dataset('../data', bench, dataset, cfg=cfg, use_all_proxies=True,
                                            cache_path=cache_path, version_key=version_key)

    if bench == 'nb201':
        ops, _ = get_ops_edges_nb201()
    elif bench == 'tnb101':    
        ops, _ = get_ops_edges_tnb101()
        ops = ops[:-1]  # no avg pooling
    elif bench == 'nb101':
        ops = get_ops_nb101()
    elif bench == 'nb301':
        ops = get_ops_nb301()
    elif bench == 'tnb101_macro':
        ops = []
    else:
        raise ValueError()

    print("Loaded: ", ops)

    better_op_names = {
        'input': 'input',
        'output': 'output',
        'none': 'zero',
        'skip_connect': 'skip',
        'nor_conv_1x1': 'C1x1',
        'nor_conv_3x3': 'C3x3',
        'sep_conv_3x3': 'SC3x3',
        'sep_conv_5x5': 'SC5x5',
        'dil_conv_3x3': 'DC3x3',
        'dil_conv_5x5': 'DC5x5',
        'maxpool3x3': 'MP3x3',
        'max_pool_3x3': 'MP3x3',
        'avg_pool_3x3': 'AP3x3',
        'conv1x1-bn-relu': 'C1x1',
        'conv3x3-bn-relu': 'C3x3'
    }

    ops = [better_op_names[o] for o in ops]
    print("More readable: ", ops)


    # In[133]:


    if bench != 'nb301':
        op_map = {str(i): k for i, k in enumerate(ops)}
    else:
        # out node is added
        op_map = {str(i + 1): k for i, k in enumerate(ops)}

    if bench == 'nb101':
        del op_map['0']
        del op_map['1']  # no input/output node


    # In[135]:


    c = 'min_path_len_banned_(0)'
    if 'min_path' in c:
        opset = eval(c.split('_banned_')[1])
        if isinstance(opset, int):
            opset = (opset,)
        inverse_set = [i for i in op_map.keys() if i not in opset]
        print(inverse_set)


    # In[158]:


    def node_degree_bench(c, bench):
        if bench in ['tnb101', 'nb201', 'nb301']:
            if 'in_degree' in c:
                return 'Input node degree - '
            if "c_{k-2}" in c:
                return 'Input 1 degree - '
            if "c_{k-1}" in c:
                return 'Input node 2 degree - '
            elif 'out_degree' in c:
                return 'Output node degree - '
            elif 'avg_in' in c:
                return 'Average out deg. - '
            elif 'avg_out' in c:
                return 'Average in deg. - '
            else:
                raise ValueError(f"Invalid node degree: {c}")
        if bench in ['nb101']:
            if 'in_degree' in c:
                return 'Output node degree - '
            elif 'out_degree' in c:
                return 'Input node degree - '
            else:
                c = c.split(')_')[1]
                assert c in ['avg_in', 'avg_out', 'max_in', 'max_out'], f"Invalid node degree: {c}"
                what, which = c.split('_')
                return f"{'Average' if what == 'avg' else 'Maximum'} {which}put node degree"


    def get_feature_name(c):
        if 'op_count' in c:
            return 'number of '
        elif 'min_path' in c:
            return 'min path over '
        elif 'max_op' in c:
            return 'max path over '
        elif 'node_degree' in c:
            return node_degree_bench(c, bench)
        else:
            raise ValueError()

    def to_better_colname(c, op_map):
        suffix = ''
        if 'normal' in c or 'reduce' in c:
            suffix = ' (normal)' if 'normal' in c else ' (reduce)'
            c = c.replace('_reduce', '').replace('_normal', '')

        if c.endswith('_c_{k-2}') or c.endswith('_c_{k-1}'):
            what = 'from input 1 ' if c == '_c_{k-2}' else 'from input 2 '
            c = c[:-8]
            suffix = f"{what}{suffix}"

        feature_name = get_feature_name(c)

        if 'min_path' in c:
            opset = eval(c.split('_banned_')[1])
            if isinstance(opset, int):
                opset = (opset,)

            opset = [str(o) for o in opset]
            opset = [i for i in op_map.keys() if i not in opset]
        elif 'op_count' in c:
            opset = c.split('_')[-1]
        else:
            opset = c.split('_allowed_')[1]
            if 'node' in c:
                opset = opset.split('_')[0]
            opset = eval(opset)

        if isinstance(opset, int):
            opset = [opset]

        opset = [op_map[str(o)] for o in opset] 
        opset = f"[{','.join(opset)}]" if len(opset) > 1 else str(opset[0])

        return f"{feature_name}{opset}{suffix}"


    def to_better_colname_macro(c):
        if 'count_ops' in c:
            vals = c.replace('count_ops_ch', '')
            channels = vals.startswith('True')  # channel
            strides = vals.endswith('True')  # downsample
        
            if not channels and not strides:
                return "number of simple convs"
        
            what = []
            if channels:
                what.append('channel increased')
            if strides:
                what.append('strided')
            return f"number of convs - {' + '.join(what)}"
        elif 'pos' in c:
            vals = c.split('_')
            assert vals[2] in ['ch', 's']
            channels = vals[2] == 'ch'
            pos = vals[-1]
            what = 'channel increases' if channels else 'strides'
            return f"Number of {what} until pos. {pos}"
        else:
            raise ValueError()


    # In[159]:


    new_cols_map = {}

    for c in data.columns:   
        if bench == 'nb301' and c in ['op_count_normal_0', 'op_count_reduce_0']:
            # skip output node
            new_cols_map[c] = c
            continue

        if bench == 'nb101' and c in ['op_count_0', 'op_count_1']:
            # skip input/output nodes
            new_cols_map[c] = c
            continue

        if bench == 'tnb101' and c == 'op_count_4':
            # included max pooling that's however not there
            new_cols_map[c] = c
            continue
        try:
            if bench != 'tnb101_macro':
                new_c = to_better_colname(c, op_map)
            else:
                new_c = to_better_colname_macro(c)

            new_cols_map[c] = new_c
        except ValueError:
            print(f'Skipping {c}')
            new_cols_map[c] = c


    # In[160]:


    print(new_cols_map)
    return new_cols_map



print("DOING MY WORK")

import os
import pandas as pd

# files = [d[2] for d in os.walk("./data/")][0] # there are only feature importances files in data at the moment
# files = [f for f in files if not f.startswith(".")] # .gitkeep

setups = [
    [ "nb101_nb301.tex", ("nb101", "cifar10", 1024), ("nb301", "cifar10", 1024)],
    [ "nb201.tex", ("nb201", "cifar10", 1024), ("nb201", "ImageNet16-120", 1024)],
    [ "tnb101.tex", ("tnb101", "autoencoder", 1024), ("tnb101", "class_scene", 1024)],
    [ "tnb101a.tex", ("tnb101", "class_object", 1024), ("tnb101", "normal", 1024)],
    [ "tnb101b.tex", ("tnb101", "jigsaw", 1024), ("tnb101", "room_layout", 1024)],
    [ "tnb101c.tex", ("tnb101", "segmentsemantic", 1024), None],
    [ "tnb101_macro.tex", ("tnb101_macro", "autoencoder", 1024), ("tnb101_macro", "class_scene", 1024)],
    [ "tnb101_macroa.tex", ("tnb101_macro", "class_object", 1024), ("tnb101_macro", "normal", 1024)],
    [ "tnb101_macrob.tex", ("tnb101_macro", "jigsaw", 1024), ("tnb101_macro", "room_layout", 1024)],
    [ "tnb101_macroc.tex", ("tnb101_macro", "segmentsemantic", 1024), None]
]


for s in setups:
    latex_name, task1, task2 = s

    dfs = [] 
    for task in task1, task2: 
        if task is None:
            continue
        print(f"Processing {task[0]} {task[1]}.")
        filename = f"{task[0]}_{task[1]}_{task[2]}.csv"
        df = pd.read_csv("data/"+filename)
        if task[0] != "tnb101_macro":
            cols_map = create_map(task[0])
            df["feature"] = df["feature"].replace(cols_map)
        
        df.columns = ["Feature name", "Mean rank"]
        dfs.append(df)

    midbreak = pd.DataFrame(index=dfs[0].index)
    midbreak["MEZERA"] = ""    

    dfs.insert(1, midbreak)
    
    df = pd.concat(dfs, axis=1)    
    with pd.option_context("max_colwidth", 1000):
        df.to_latex(latex_name, index=False)


