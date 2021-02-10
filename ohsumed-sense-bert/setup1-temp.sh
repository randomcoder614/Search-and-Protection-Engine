#!/bin/bash

#js=(1 3 4 6 9 10)
js=(4)

cd ..

echo "Training models with training measure: NDCG"
for j in "${js[@]}";
do
	echo "--------------------------- fold = ${i}, algorithm = ${j}"
	java  -Xmx4g -jar RankLib-2.1.jar -train OHSUMED-BERT/Fold5/train.txt -validate OHSUMED-BERT/Fold5/vali.txt -ranker ${j} -metric2t NDCG@10 -save ohsumed-sense-bert/models1/${j}f5k1-ndcg.model -sFile ohsumed-labels-probabilities.csv
done  


echo "Ranking results"
for i in `seq 1 5`;
do
	for j in "${js[@]}";
	do
		java -jar RankLib-2.1.jar -load ohsumed-sense-bert/models1/${j}f${i}k1-ndcg.model -rank OHSUMED-BERT/Fold${i}/test.txt -indri ohsumed-sense-bert/output1/indri-${j}f${i}k1.txt -sFile ohsumed-labels-probabilities.csv
	done
done

for j in "${js[@]}";
do
	cat ohsumed-sense-bert/output1/indri-${j}f2k1.txt ohsumed-sense-bert/output1/indri-${j}f3k1.txt ohsumed-sense-bert/output1/indri-${j}f4k1.txt ohsumed-sense-bert/output1/indri-${j}f5k1.txt ohsumed-sense-bert/output1/indri-${j}f1k1.txt > ohsumed-sense-bert/output1/indri-${j}allk1.txt 
done



