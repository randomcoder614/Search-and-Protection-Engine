#!/bin/bash

#js=(1 3 4 6 9 10)
js=(4)

cd ..


echo "Training models with training measure: NDCG"
for i in `seq 1 5`;
do
	for j in "${js[@]}";
	do
		echo "--------------------------- fold = ${i}, algorithm = ${j}"
		java -jar RankLib-2.1.jar -train OHSUMED/QueryLevelNorm/Fold${i}/train.filtered.txt -validate OHSUMED/QueryLevelNorm/Fold${i}/vali.filtered.txt -ranker ${j} -metric2t NDCG@10 -save ohsumed-sense/models2/${j}f${i}k1-ndcg.model -sFile ohsumed-labels-probabilities.csv
	done  
done  

echo "Ranking results"
for i in `seq 1 5`;
do
	for j in "${js[@]}";
	do
		java -jar RankLib-2.1.jar -applysensepred -load ohsumed-sense/models2/${j}f${i}k1-ndcg.model -rank OHSUMED/QueryLevelNorm/Fold${i}/test.txt -indri ohsumed-sense/output2/indri-${j}f${i}k1.txt -sFile ohsumed-labels-probabilities.csv
	done  
done  

for j in "${js[@]}";
do
	cat ohsumed-sense/output2/indri-${j}f2k1.txt ohsumed-sense/output2/indri-${j}f3k1.txt ohsumed-sense/output2/indri-${j}f4k1.txt ohsumed-sense/output2/indri-${j}f5k1.txt ohsumed-sense/output2/indri-${j}f1k1.txt > ohsumed-sense/output2/indri-${j}allk1.txt 
done


#ks=(3 6 9 15 30 60 90)
#ks=(3 6 15 100 1000 10000)
ks=(0 1 3 5)
echo "Testing models with different evaluation measures"
for k in "${ks[@]}";
do
	for i in `seq 1 5`;
	do
		for j in "${js[@]}";
		do
			java -jar RankLib-2.1.jar -applysensepred -sensek ${k} -load ohsumed-sense/models2/${j}f${i}k1-ndcg.model -test OHSUMED/QueryLevelNorm/Fold${i}/test.txt -metric2T NDCG@10 -idv ohsumed-sense/output2/${j}f${i}k${k}-ndcg.ndcg.txt -sFile ohsumed-labels-probabilities.csv
			java -jar RankLib-2.1.jar -applysensepred -sensek ${k} -load ohsumed-sense/models2/${j}f${i}k1-ndcg.model -test OHSUMED/QueryLevelNorm/Fold${i}/test.txt -metric2T SENSE@10 -idv ohsumed-sense/output2/${j}f${i}k${k}-ndcg.sense.txt -sFile ohsumed-labels-probabilities.csv 
		done  
	done  
done
