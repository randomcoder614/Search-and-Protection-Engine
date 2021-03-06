#!/bin/bash

#ks=(3 6 9 15 30 60 90)
#ks=(12 15 18)
#ks=(12)
#ks=(3 6 15 100 1000 10000)
#ks=(3 15 10000)
#ks=(1 5 10)
#ks=(1)
ks=(15)

#js=(1 3 4 6 9 10)
js=(4)

cd ..

echo "Training models with training measures: ALPHACS_NDCG@10"
for k in "${ks[@]}";
do
	for i in `seq 1 5`;
	do
		for j in "${js[@]}";
		do
			echo "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx sensek = ${k}, fold = ${i}, algorithm = ${j}"
			java -jar RankLib-2.1.jar -sensek ${k} -train OHSUMED/QueryLevelNorm/Fold${i}/train.withprob.txt -validate OHSUMED/QueryLevelNorm/Fold${i}/vali.withprob.txt -ranker ${j} -metric2t ALPHACS_NDCG@10 -save ohsumed-sense/models4/${j}f${i}k${k}-ncs_dcg.model -sFile ohsumed-labels-probabilities.csv
		done  
	done  
done

echo "Ranking results"
for k in "${ks[@]}";
do
	for i in `seq 1 5`;
	do
		for j in "${js[@]}";
		do
			java -jar RankLib-2.1.jar -load ohsumed-sense/models4/${j}f${i}k${k}-ncs_dcg.model -rank OHSUMED/QueryLevelNorm/Fold${i}/test.withprob.txt -indri ohsumed-sense/output4/indri-ncs_dcg-${j}f${i}k${k}.txt -sFile ohsumed-labels-probabilities.csv
			java -jar RankLib-2.1.jar -applysensepred -load ohsumed-sense/models4/${j}f${i}k${k}-ncs_dcg.model -rank OHSUMED/QueryLevelNorm/Fold${i}/test.withprob.txt -indri ohsumed-sense/output4/indri-ncs_dcg-filtered-${j}f${i}k${k}.txt -sFile ohsumed-labels-probabilities.csv
		done  
	done  
done  

for k in "${ks[@]}";
do
	for j in "${js[@]}";
	do
		cat ohsumed-sense/output4/indri-ncs_dcg-${j}f2k${k}.txt ohsumed-sense/output4/indri-ncs_dcg-${j}f3k${k}.txt ohsumed-sense/output4/indri-ncs_dcg-${j}f4k${k}.txt ohsumed-sense/output4/indri-ncs_dcg-${j}f5k${k}.txt ohsumed-sense/output4/indri-ncs_dcg-${j}f1k${k}.txt > ohsumed-sense/output4/indri-ncs_dcg-${j}allk${k}.txt 
		cat ohsumed-sense/output4/indri-ncs_dcg-filtered-${j}f2k${k}.txt ohsumed-sense/output4/indri-ncs_dcg-filtered-${j}f3k${k}.txt ohsumed-sense/output4/indri-ncs_dcg-filtered-${j}f4k${k}.txt ohsumed-sense/output4/indri-ncs_dcg-filtered-${j}f5k${k}.txt ohsumed-sense/output4/indri-ncs_dcg-filtered-${j}f1k${k}.txt > ohsumed-sense/output4/indri-ncs_dcg-filtered-${j}allk${k}.txt 
	done
done  

