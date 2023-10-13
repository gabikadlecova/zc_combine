. "$1"/bin/activate

if [ -n $3 ]; then
  cache_dir="--cache_dir_ $3"
fi

if [ -n $4 ]; then
  version_key="--version_key $4"
fi

features_out=`python compute_pca.py  $pca_args $cache_dir $version_key`

if [ -n $row_id ]; then
  SUFFIX="-$row_id"
fi

for name in "" "_train"; do
  out_path=`python feature_selection.py --imp_path "$features_out/pca$name.csv" $feature_args`

  # connect feature selection args together with _ and - (for logging)
  PREFIX=`echo $feature_args | sed 's/--\([a-zA-Z0-1]*\) /\1_/g' | sed 's/ /-/g'`

  python train_on_features.py --columns_json_ $out_path --wandb_key_ $2 $train_args $cache_dir \
      --args_json_ "$features_out/args.json" --out_prefix pca"$name"-"$PREFIX""$SUFFIX"
done

cat "$features_out/args.json" > /dev/null && rm -rf $features_out
