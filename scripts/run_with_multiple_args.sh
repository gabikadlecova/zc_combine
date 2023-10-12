#!/bin/bash

# - run pca on with on every $3th config (shifted by $2)
# - configs are in a .csv file ($1), generated by create_args_df.py

infile=$1
id=$2
step=$3
shift 3

python get_script_args.py --input_file $infile --id $id --step $step | while read row; do
  echo $row
  eval $row; $@
done