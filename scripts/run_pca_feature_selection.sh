. "$1"/bin/activate

features_out=`python compute_pca.py  $PCA_ARGS`

for name in "" "_train"; do
  out_path=`python feature_selection.py --imp_path "$features_out/pca$name.csv" $FEATURE_ARGS`

  # connect feature selection args together with _ and - (for logging)
  PREFIX=`echo $FEATURE_ARGS | sed 's/--\([a-zA-Z0-1]*\) /\1_/g' | sed 's/ /-/g'`

  python train_on_features.py --columns_json_ $out_path --wandb_key_ $2 $TRAIN_ARGS \
      --args_json_ "$features_out/args.json" --out_prefix pca"$name"-"$PREFIX"
done


python train_on_features.py --wandb_key_ $2 $TRAIN_ARGS \
        --args_json_ "$features_out/args.json"

cat "$features_out/args.json" > /dev/null && rm -rf $features_out
