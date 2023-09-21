. "$1"/bin/activate

if [ -n "$4" ]; then
    echo "$4"
    da_path="--columns_json $4"
    base_dir=`basename "$4"`
    da_args=`python parse_args.py "$base_dir"`
fi

python train_on_features.py --benchmark $2 --cfg ../zc_combine/configs/"$2"_first.json --meta ../data/meta.json \
  --model $3 --wandb_key f98bc29c22b4f6cf73d51153c2712262dfc60fee $da_path $da_args
