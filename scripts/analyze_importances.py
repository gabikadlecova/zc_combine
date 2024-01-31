
import os
import json
import sys

import pandas as pd

TRAIN_SIZE=int(sys.argv[1])
BENCHMARK=sys.argv[2]
DATASET=sys.argv[3]

runs = [d[0] for d in os.walk(".") if d[0].startswith("./train")] 
#print(runs)

def load_args(run):
    with open(f"{run}/args.json") as f:
        args = json.load(f)
    return args 

def load_importances(run, i):
    imp = (
        pd.read_csv(f"{run}/imp_{i}.csv")
        .drop(columns=["seed"])
        .T
        .reset_index()  
    )

    imp.columns = ["feature", "importance"]
    
    imp = (
        imp
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
        .assign(rank=imp.index)
        .set_index("feature")
        .sort_index()
     )    
    return imp

def analyze_importances(run):
    importances = []
    for i in range(42,92):
        ip_df = load_importances(run, i)[["rank"]]
    #    ip_df = ip_df[ip_df["rank"] < 20]

        importances.append(ip_df)
        
        
    importances = (
         pd.concat(importances, axis=1)
        .T
        .describe()
        .T["mean"]  # mean rank
        .sort_values()
    )


    print(importances[:10])
    importances.name = "mean rank"
    importances[:10].to_csv(f"{BENCHMARK}_{DATASET}_{TRAIN_SIZE}.csv")
    #    print(importances.dropna())


    
for run in runs:
    args = load_args(run)


    if (
            args["benchmark"] !=  BENCHMARK or
            args["train_size"] != TRAIN_SIZE or
            args["dataset"] != DATASET or
            args["use_all_proxies"] != True or
            args["use_features"] != True or
            args["use_onehot"] != False or
            args["use_flops_params"] not in (False, None) or
            args["use_path_encoding"] != False
    ):
        continue

    analyze_importances(run)
    
