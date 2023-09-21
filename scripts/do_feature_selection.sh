. "$1"/bin/activate


s1='--mode norm --n_features 10'
s2='--mode norm --n_features 20'
s3='--mode norm --n_features 30'

for s in "$s1" "$s2" "$s3"; do
  out_path=`python feature_selection.py --imp_path "$2" --out_prefix "$3" $s`
  echo $out_path
done
