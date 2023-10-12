. "$1"/bin/activate

features_out=`python compute_pca.py  $pca_args`

if [ -n $row_id ]; then
  SUFFIX="-$row_id"
fi

for name in "" "_train"; do
  out_path=`python feature_selection.py --imp_path "$features_out/pca$name.csv" $feature_args`

  # connect feature selection args together with _ and - (for logging)
  PREFIX=`echo $feature_args | sed 's/--\([a-zA-Z0-1]*\) /\1_/g' | sed 's/ /-/g'`

  python train_on_features.py --columns_json_ $out_path --wandb_key_ $2 $train_args \
      --args_json_ "$features_out/args.json" --out_prefix pca"$name"-"$PREFIX""$SUFFIX"
done


python train_on_features.py --wandb_key_ $2 $train_args \
        --args_json_ "$features_out/args.json"

cat "$features_out/args.json" > /dev/null && rm -rf $features_out
