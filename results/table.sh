echo "" > important_features.tex

for FILE in nb101_nb301 nb201 tnb101 tnb101a tnb101b tnb101c  tnb101_macro tnb101_macroa tnb101_macrob tnb101_macroc 
do
    echo "\begin{table}" >> important_features.tex
    echo "\caption{$FILE}" >> important_features.tex
    cat $FILE.tex | sed 's|MEZERA|\$\\quad\$|g' | sed 's|toprule|hline|g' | sed 's|bottomrule|hline|g' | sed 's|midrule|hline|g' >> important_features.tex
    echo "\end{table}" >> important_features.tex
    echo "" >> important_features.tex
    echo "" >> important_features.tex
    echo "" >> important_features.tex
    
done 
