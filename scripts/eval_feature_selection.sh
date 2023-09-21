. "$1"/bin/activate

run_features() {
  if [ -n "$3" ]; then
    echo "$3"
    da_path="--columns_json $3"
    base_dir=`dirname "$3"`
    da_args=`python parse_args.py "$base_dir/args.json"`
    echo $da_args
  fi

  python train_on_features.py --benchmark $1 --cfg ../zc_combine/configs/"$1"_first.json --meta ../data/meta.json \
    --model $2 --wandb_key f98bc29c22b4f6cf73d51153c2712262dfc60fee $da_path $da_args
}

for file in feat_sel_files_proxy/*"$2"*_train_size-"$3"_*; do
    for pca_file in `cat $file`; do
        run_features $2 $4 "$pca_file"
    done
done
