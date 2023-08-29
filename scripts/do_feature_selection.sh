`. "$1"/bin/activate`

s1='--mode idx --idx 0 --n_features 20'
s2='--mode mean --n_features 20'
s3='--mode row --n_features 10'
s4='--mode row --n_features 20'


for s in "$s1" "$s2" "$s3" "$s4"; do
  echo "$s"
  out_path=`python feature_selection.py --imp_path "$2" --out_prefix "$3" $s`
  echo $out_path
done
