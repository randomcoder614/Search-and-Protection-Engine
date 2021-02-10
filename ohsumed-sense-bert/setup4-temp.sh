#!/bin/bash

#ks=(3 6 9 15 30 60 90)
#ks=(12 15 18)
#ks=(12)
#ks=(3 6 15 100 1000 10000)
#ks=(3 15 10000)
#ks=(1 5 10)
#ks=(1)
ks=(1)

#js=(1 3 4 6 9 10)
js=(4)

cd ..

echo "Training models with training measures: SENSE"
for k in "${ks[@]}";
do
	for j in "${js[@]}";
	do
		echo "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx sensek = ${k}, fold = ${i}, algorithm = ${j}"
		java -Xmx4g -jar RankLib-2.1.jar -sensek ${k} -train OHSUMED-BERT/Fold5/train.withprob.txt -validate OHSUMED-BERT/Fold5/vali.withprob.txt -ranker ${j} -metric2t SENSE@10 -save ohsumed-sense-bert/models4/${j}f5k${k}-sense.model -sFile ohsumed-labels-probabilities.csv
	done  
done
