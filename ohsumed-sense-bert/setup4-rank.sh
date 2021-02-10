#!/bin/bash

#ks=(3 6 9 15 30 60 90)
#ks=(12 15 18)
#ks=(12)
#ks=(3 6 15 100 1000 10000)
#ks=(3 15 10000)
#ks=(1 5 10)
#ks=(1)
ks=(0 1 3 5)

#js=(1 3 4 6 9 10)
js=(4)

cd ..


echo "Ranking results"
for k in "${ks[@]}";
do
	for i in `seq 1 5`;
	do
		for j in "${js[@]}";
		do
			java -jar RankLib-2.1.jar -load ohsumed-sense-bert/models4/${j}f${i}k${k}-sense.model -rank OHSUMED-BERT/Fold${i}/test.withprob.txt -indri ohsumed-sense-bert/output4/indri-${j}f${i}k${k}.txt -sFile ohsumed-labels-probabilities.csv
			java -jar RankLib-2.1.jar -applysensepred -load ohsumed-sense-bert/models4/${j}f${i}k${k}-sense.model -rank OHSUMED-BERT/Fold${i}/test.withprob.txt -indri ohsumed-sense-bert/output4/indri-filtered-${j}f${i}k${k}.txt -sFile ohsumed-labels-probabilities.csv
		done  
	done  
done  

for k in "${ks[@]}";
do
	for j in "${js[@]}";
	do
		cat ohsumed-sense-bert/output4/indri-${j}f2k${k}.txt ohsumed-sense-bert/output4/indri-${j}f3k${k}.txt ohsumed-sense-bert/output4/indri-${j}f4k${k}.txt ohsumed-sense-bert/output4/indri-${j}f5k${k}.txt ohsumed-sense-bert/output4/indri-${j}f1k${k}.txt > ohsumed-sense-bert/output4/indri-${j}allk${k}.txt 
		cat ohsumed-sense-bert/output4/indri-filtered-${j}f2k${k}.txt ohsumed-sense-bert/output4/indri-filtered-${j}f3k${k}.txt ohsumed-sense-bert/output4/indri-filtered-${j}f4k${k}.txt ohsumed-sense-bert/output4/indri-filtered-${j}f5k${k}.txt ohsumed-sense-bert/output4/indri-filtered-${j}f1k${k}.txt > ohsumed-sense-bert/output4/indri-filtered-${j}allk${k}.txt 
	done
done

