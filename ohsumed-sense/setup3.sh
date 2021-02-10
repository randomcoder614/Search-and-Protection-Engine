#!/bin/bash

#js=(1 3 4 6 9 10)
js=(4 10)

cd ..

echo "Ranking results"
for i in `seq 1 5`;
do
	for j in "${js[@]}";
	do
		java -jar RankLib-2.1.jar -applysensepred -load ohsumed-sense/models1/${j}f${i}k1-ndcg.model -rank OHSUMED/QueryLevelNorm/Fold${i}/test.txt -indri ohsumed-sense/output3/indri-${j}f${i}k1.txt -sFile ohsumed-labels-probabilities.csv
	done  
done  

for j in "${js[@]}";
do
	cat ohsumed-sense/output3/indri-${j}f2k1.txt ohsumed-sense/output3/indri-${j}f3k1.txt ohsumed-sense/output3/indri-${j}f4k1.txt ohsumed-sense/output3/indri-${j}f5k1.txt ohsumed-sense/output3/indri-${j}f1k1.txt > ohsumed-sense/output3/indri-${j}allk1.txt 
done

