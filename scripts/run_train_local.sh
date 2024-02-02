. "$1"/bin/activate


if [ -n "$3" ]; then
  cache_dir="--cache_dir_ $3"
fi

if [ -n "$4" ]; then
  version_key="--version_key $4"
fi

if [ -n "$row_id" ]; then
  SUFFIX="-$row_id"
fi

python train_on_features.py --out_ "$2" $train_args --out_prefix train_"$SUFFIX" $cache_dir $version_key

